import os
import json
import time
import hashlib
from pathlib import Path
from typing import List, Optional, Dict, Tuple, Any

import numpy as np
import torch
from torch.utils.data import Dataset
from torchvision.transforms import functional
from torchvision.transforms import functional as F
from PIL import Image
import cv2
import matplotlib.pyplot as plt


# =============== 設定 ===============
# IMAGES_DIR = Path('../../kuzushiji-recognition/char_sep_datas')  # 画像ディレクトリ
# GT_JSON_PATH = Path('../../kuzushiji-recognition/char_sep_datas/gt_json.json')    # アノテーションJSON (未使用でもロード例)

# =============== Dataset 最小実装 ===============
class FineTuningDataset_v1_1(Dataset):
    """
    Fine tuning 用 Dataset 高速化版

    機能:
      - 画像読み込み & 横幅固定リサイズ (アスペクト比保持)
      - 4チャネル GT ヒートマップ生成（main / affinity / furi / furi_affinity）
      - オンザフライ or 事前計算 (precompute_gt)
      - ディスクキャッシュ (cache_dir) + メモリキャッシュ (LRU)
      - FP16 (use_fp16_gt) によるメモリ削減
      - ベンチマークユーティリティ

    注意: 空間的 (RandomCrop / Flip 等) 変換を transform に入れる場合、
          precompute_gt=True だと input と GT がズレるため False 推奨。
    """

    def __init__(
        self,
        test_doc_id: List[str],
        test_mode: bool = False,
        images_dir:  Path = Path('../kuzushiji-recognition/char_sep_datas'),
        json_path: Path | None = Path('../kuzushiji-recognition/char_sep_datas/gt_json.json'),
        extensions: Optional[List[str]] = None,
        transform=None,
        target_width: int = 300,
        # 追加高速化オプション
        precompute_gt: bool = False,
        cache_dir: str | Path | None = '.cache/fine_v1_gt',
        use_disk_cache: bool = True,
        in_memory_gt: bool = True,
        use_fp16_gt: bool = False,
        image_cache_size: int = 64,
        allow_cache_with_transform: bool = False,
        verbose: bool = True,
        # ---- ガウス中心バイアス制御 ----
        center_bias_alpha: float = 0.0,  # 0.0=オフ, 1.0=完全にターゲット側へ
        center_bias_mode: str = 'toward_long_base',  # 'toward_long_base' | 'toward_short_base' | 'custom_fraction'
        center_bias_fraction: float = 0.3,  # mode='custom_fraction' の時 0=上 1=下 (長辺回転後の矩形座標)
    ):
        self.test_doc_id = test_doc_id
        self.test_mode = test_mode
        self.images_dir = Path(images_dir)
        self.transform = transform
        self.extensions = extensions or ['.jpg', '.jpeg', '.png']
        self.target_width = int(target_width)
        self.precompute_gt = precompute_gt
        self.use_disk_cache = use_disk_cache
        self.in_memory_gt = in_memory_gt
        self.use_fp16_gt = use_fp16_gt
        self.image_cache_size = max(0, image_cache_size)
        self.allow_cache_with_transform = allow_cache_with_transform
        self.verbose = verbose
        # ガウス中心バイアス設定
        self.center_bias_alpha = float(center_bias_alpha)
        self.center_bias_mode = center_bias_mode
        self.center_bias_fraction = float(center_bias_fraction)

        try:
            import cv2  # noqa: F401
            self._HAS_CV2 = True
        except ImportError:
            self._HAS_CV2 = False

        assert self.images_dir.exists(), f"画像ディレクトリが存在しません: {self.images_dir}"

        # 画像パス列挙
        self.image_paths: List[Path] = self._get_image_path()
        if not self.image_paths:
            raise RuntimeError(f"画像が見つかりません: {self.images_dir}")

        # JSON 読み込み
        self.raw_json: Optional[Dict[str, Any]] = None
        if json_path is not None:
            jp = Path(json_path)
            if jp.exists():
                with open(jp, 'r', encoding='utf-8') as f:
                    self.raw_json = json.load(f)

        # キャッシュ用
        self.cache_dir = Path(cache_dir) if cache_dir is not None else None
        if self.cache_dir and self.use_disk_cache:
            self.cache_dir.mkdir(parents=True, exist_ok=True)

        # メモリ GT / 画像キャッシュ
        self._gt_cache_mem: Dict[str, torch.Tensor] = {}
        self._img_cache: Dict[str, torch.Tensor] = {}
        self._img_cache_order: List[str] = []  # LRU

        # transform が空間変換含む場合 (簡易判定) & precompute 要求 → 無効化
        if self.precompute_gt and self.transform and not self.allow_cache_with_transform:
            text = repr(self.transform)
            risky_keywords = ['Random', 'Crop', 'Flip', 'Affine']
            if any(k in text for k in risky_keywords):
                if self.verbose:
                    print('[FineTuningDataset_v1] 空間変換を検出したため precompute_gt を自動無効化')
                self.precompute_gt = False

        if self.precompute_gt:
            t0 = time.time()
            self._precompute_all_gt()
            if self.verbose:
                dt = time.time() * 1000 - t0 * 1000
                print(f'[FineTuningDataset_v1] precompute 完了 in {dt:.1f} ms  count={len(self._gt_cache_mem)}')

    def _get_image_path(self):
        images_path = []
        for doc_path in self.images_dir.iterdir():
            if not doc_path.is_file():
                cond_A = self.test_mode
                cond_B = doc_path.name in self.test_doc_id
                test_cond = cond_A and cond_B
                train_cond = not (cond_A or cond_B)
                if test_cond or train_cond:
                    images_path_obj = doc_path / 'images'
                    for image_obj in images_path_obj.iterdir():
                        images_path.append(image_obj)
        return images_path

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx: int):
        path = self.image_paths[idx]

        # 画像ロード + リサイズ + Tensor 化 (キャッシュ利用)
        img_tensor, resized_pil, original_size = self._load_image_tensor(path, cache=True)

        # GT 取得
        doc_id, image_id = self._get_doc_image_id(path)
        key = self._cache_key(doc_id, image_id)
        gt_tensor = self._get_or_build_gt(key, resized_pil, original_size, doc_id, image_id)

        return img_tensor, gt_tensor

    # --------------------------------------------------
    # キャッシュキー
    # --------------------------------------------------
    def _cache_key(self, doc_id: str, image_id: str) -> str:
        return f'{doc_id}_sep_{image_id}'

    def _gt_file_path(self, key: str) -> Path:
        assert self.cache_dir is not None
        h = hashlib.md5(key.encode('utf-8')).hexdigest()
        return self.cache_dir / f'{h}.pt'

    # --------------------------------------------------
    # 画像ロード & リサイズ & Tensor 変換 (LRU キャッシュ)
    # --------------------------------------------------
    def _load_image_tensor(self, path: Path, cache: bool = True) -> Tuple[torch.Tensor, Image.Image, Tuple[int, int]]:
        path_str = str(path)
        if cache and self.image_cache_size > 0 and path_str in self._img_cache:
            tensor, resized_pil, orig = self._img_cache[path_str]
            # LRU 更新
            self._img_cache_order.remove(path_str)
            self._img_cache_order.append(path_str)
            return tensor, resized_pil, orig

        image = Image.open(path).convert('RGB')
        original_w, original_h = image.size
        aspect_ratio = original_h / max(original_w, 1)
        new_w = self.target_width
        new_h = int(round(self.target_width * aspect_ratio))
        new_size = (new_h, new_w)
        resized = functional.resize(image, new_size, interpolation=functional.InterpolationMode.BILINEAR)

        if self.transform:
            tensor = self.transform(resized)
        else:
            tensor = F.to_tensor(resized)

        if cache and self.image_cache_size > 0:
            self._img_cache[path_str] = (tensor, resized, (original_w, original_h))
            self._img_cache_order.append(path_str)
            # LRU 追い出し
            if len(self._img_cache_order) > self.image_cache_size:
                old = self._img_cache_order.pop(0)
                self._img_cache.pop(old, None)

        return tensor, resized, (original_w, original_h)

    # --------------------------------------------------
    # GT 取得 (メモリ→ディスク→生成)
    # --------------------------------------------------
    def _get_or_build_gt(self, key: str, image_pil: Image.Image, original_size: Tuple[int, int], doc_id: str, image_id: str) -> torch.Tensor:
        # 1. メモリ
        if self.in_memory_gt and key in self._gt_cache_mem:
            # print('[FineTuningDataset_v1] GT メモリキャッシュ ヒット:', key)
            return self._gt_cache_mem[key]
        # 2. ディスク
        if self.use_disk_cache and self.cache_dir is not None:
            fp = self._gt_file_path(key)
            if fp.exists():
                try:
                    gt = torch.load(fp, map_location='cpu')
                    if self.in_memory_gt:
                        self._gt_cache_mem[key] = gt
                    # print('[FineTuningDataset_v1] GT ディスクキャッシュ ヒット:', key)
                    return gt
                except Exception:
                    pass  # 壊れていたら再生成
        # 3. 生成
        gt_info = self._get_gt_info(doc_id, image_id)
        gt = self.generate_gt_heatmaps(gt_info, image_pil, original_size=original_size)
        if self.use_fp16_gt:
            gt = gt.half()
        # 保存
        if self.in_memory_gt:
            self._gt_cache_mem[key] = gt
        if self.use_disk_cache and self.cache_dir is not None:
            try:
                torch.save(gt, self._gt_file_path(key))
            except Exception:
                pass
        # print('[FineTuningDataset_v1] GT 生成:', key)
        return gt

    # --------------------------------------------------
    # 全件 precompute
    # --------------------------------------------------
    def _precompute_all_gt(self):
        for path in self.image_paths:
            doc_id, image_id = self._get_doc_image_id(path)
            key = self._cache_key(doc_id, image_id)
            if self.in_memory_gt and key in self._gt_cache_mem:
                continue
            if self.use_disk_cache and self.cache_dir is not None and self._gt_file_path(key).exists():
                if self.in_memory_gt:
                    try:
                        gt_loaded = torch.load(self._gt_file_path(key), map_location='cpu')
                        self._gt_cache_mem[key] = gt_loaded
                    except Exception:
                        pass
                continue
            # 画像 (リサイズ含む) 取得 (画像キャッシュ使う)
            _, resized_pil, original_size = self._load_image_tensor(path, cache=True)
            gt_info = self._get_gt_info(doc_id, image_id)
            gt = self.generate_gt_heatmaps(gt_info, resized_pil, original_size=original_size)
            if self.use_fp16_gt:
                gt = gt.half()
            if self.in_memory_gt:
                self._gt_cache_mem[key] = gt
            if self.use_disk_cache and self.cache_dir is not None:
                try:
                    torch.save(gt, self._gt_file_path(key))
                except Exception:
                    pass
    
    def _get_doc_image_id(self, path: Path) -> Tuple[str, str]:
        parts = path.parts
        # .../doc_id/images/xxxxx.png を想定
        if len(parts) < 3:
            return 'unknown', path.stem
        doc_id = parts[-3]
        image_id = path.stem
        return doc_id, image_id
    
    def _get_gt_info(self, doc_id: str, image_id: str) -> Dict[str, list]:
        key = f"{doc_id}_sep_{image_id}"
        empty = {
            'main_region': [],
            'main_affinity': [],
            'furi_region': [],
            'furi_affinity': []
        }
        if not self.raw_json or 'files' not in self.raw_json:
            return empty
        return self.raw_json['files'].get(key, empty)
        

    def _add_perspective_gaussian(self, canvas, quad_points, amplitude=1.0, min_gaussian_size=5):
        """
        quad_points: [(x0,y0),(x1,y1),(x2,y2),(x3,y3)]
        canvas: np.ndarray (H,W) float32
        台形の重心に対応する矩形座標をガウス中心に置き、矩形→台形へ warp して加算。
        """
        if not self._HAS_CV2:
            # OpenCVなしフォールバック: 外接矩形に2Dガウシアンをそのまま加算
            xs = [p[0] for p in quad_points]
            ys = [p[1] for p in quad_points]
            x0, x1 = int(np.floor(min(xs))), int(np.ceil(max(xs)))
            y0, y1 = int(np.floor(min(ys))), int(np.ceil(max(ys)))
            if x1 <= x0 or y1 <= y0:
                return
            w = max(x1 - x0, 1)
            h = max(y1 - y0, 1)
            # 最小サイズ確保
            w = max(w, min_gaussian_size)
            h = max(h, min_gaussian_size)
            gx = np.linspace(-w / 2, w / 2, w)
            gy = np.linspace(-h / 2, h / 2, h)
            gx, gy = np.meshgrid(gx, gy)
            sigma_x = max(w / 5.0, 1.0)
            sigma_y = max(h / 5.0, 1.0)
            g = amplitude * np.exp(-((gx ** 2) / (2 * sigma_x ** 2) + (gy ** 2) / (2 * sigma_y ** 2))).astype(np.float32)
            # クリップ範囲
            x1_clip = min(x0 + w, canvas.shape[1])
            y1_clip = min(y0 + h, canvas.shape[0])
            canvas[y0:y1_clip, x0:x1_clip] += g[: y1_clip - y0, : x1_clip - x0]
            return

        def _order_clockwise(pts: np.ndarray) -> np.ndarray:
            # 4点を重心基準の角度で時計回り整列（開始点は任意）
            c = pts.mean(axis=0)
            ang = np.arctan2(pts[:, 1] - c[1], pts[:, 0] - c[0])
            order = np.argsort(ang)
            return pts[order]

        def _align_long_base_as_horizontal(pts_cw: np.ndarray) -> np.ndarray:
            """
            pts_cw: 時計回り [p0,p1,p2,p3]
            対向辺 (0-1,2-3) と (1-2,3-0) の合計長を比較し、
            長い「平行辺ペア」を (0-1,2-3) に来るように回転。
            """
            def edgelens(q):
                l01 = np.linalg.norm(q[0] - q[1])
                l12 = np.linalg.norm(q[1] - q[2])
                l23 = np.linalg.norm(q[2] - q[3])
                l30 = np.linalg.norm(q[3] - q[0])
                return l01, l12, l23, l30

            l01, l12, l23, l30 = edgelens(pts_cw)
            sum_base_A = l01 + l23   # 対向辺ペア (0-1, 2-3)
            sum_base_B = l12 + l30   # 対向辺ペア (1-2, 3-0)
            if sum_base_B > sum_base_A:
                # 1つ回転して (1-2,3-0) を (0-1,2-3) に合わせる
                pts_cw = np.roll(pts_cw, shift=-1, axis=0)
            return pts_cw

        try:
            # 入力点を整形
            src_raw = np.array(quad_points, dtype=np.float32)
            # 1) 時計回り整列 → 2) 長い平行辺ペアが (0-1,2-3) になるよう回転
            src = _order_clockwise(src_raw)
            src = _align_long_base_as_horizontal(src).astype(np.float32)

            # 対向辺の平均長で幅・高さを安定化
            w_top = np.linalg.norm(src[0] - src[1])
            w_bot = np.linalg.norm(src[2] - src[3])
            h_left = np.linalg.norm(src[0] - src[3])
            h_right = np.linalg.norm(src[1] - src[2])
            width = int(max(1, round((w_top + w_bot) / 2.0)))
            height = int(max(1, round((h_left + h_right) / 2.0)))

            # 最小サイズ確保
            width = max(width, min_gaussian_size)
            height = max(height, min_gaussian_size)

            # 矩形頂点（x,y）: (0,0)→(w-1,0)→(w-1,h-1)→(0,h-1)
            dst = np.array(
                [[0, 0], [width - 1, 0], [width - 1, height - 1], [0, height - 1]],
                dtype=np.float32,
            )

            # 透視変換行列（矩形→台形）とその逆行列
            M = cv2.getPerspectiveTransform(dst, src)
            Minv = np.linalg.inv(M)

            # 台形の重心（多角形重心）を計算
            def _polygon_centroid(pts: np.ndarray) -> Tuple[float, float]:
                x = pts[:, 0]
                y = pts[:, 1]
                x2 = np.append(x, x[0])
                y2 = np.append(y, y[0])
                cross = x2[:-1] * y2[1:] - x2[1:] * y2[:-1]
                A = cross.sum() / 2.0
                if abs(A) < 1e-6:
                    return float(x.mean()), float(y.mean())
                Cx = ((x2[:-1] + x2[1:]) * cross).sum() / (6.0 * A)
                Cy = ((y2[:-1] + y2[1:]) * cross).sum() / (6.0 * A)
                return float(Cx), float(Cy)

            cx_src, cy_src = _polygon_centroid(src.astype(np.float64))

            # 重心（台形座標）を矩形座標へ逆写像 → ガウス基準中心
            pt = np.array([[[cx_src, cy_src]]], dtype=np.float32)
            cx_rect, cy_rect = cv2.perspectiveTransform(pt, Minv)[0, 0]

            # ---- 中心バイアス（長辺 or 短辺 方向へシフト）----
            if self.center_bias_alpha > 0.0:
                # w_top, w_bot は既に計算済み
                long_top = w_top >= w_bot
                # ターゲット Y 決定 (矩形座標 0=上, height-1=下)
                if self.center_bias_mode == 'toward_long_base':
                    target_y = 0 if long_top else (height - 1)
                elif self.center_bias_mode == 'toward_short_base':
                    target_y = 0 if not long_top else (height - 1)
                elif self.center_bias_mode == 'custom_fraction':
                    # 0=上,1=下 に線形にマップ
                    frac = min(max(self.center_bias_fraction, 0.0), 1.0)
                    target_y = frac * (height - 1)
                else:
                    target_y = cy_rect  # 未知モード→無効
                # 線形内挿 (alpha=1 なら完全ターゲット)
                cy_rect = (1.0 - self.center_bias_alpha) * cy_rect + self.center_bias_alpha * target_y
                # クランプ
                cy_rect = float(np.clip(cy_rect, 0.0, height - 1.0))

            # ガウス生成（矩形画像上）
            xg = np.arange(width, dtype=np.float32)
            yg = np.arange(height, dtype=np.float32)
            gx, gy = np.meshgrid(xg, yg)
            sigma_x = max(width / 5.0, 1.0)
            sigma_y = max(height / 5.0, 1.0)
            gaussian = amplitude * np.exp(
                -(((gx - cx_rect) ** 2) / (2 * sigma_x ** 2) + ((gy - cy_rect) ** 2) / (2 * sigma_y ** 2))
            ).astype(np.float32)

            # 矩形→台形にワープしてキャンバスへ加算
            warped = cv2.warpPerspective(
                gaussian,
                M,
                (canvas.shape[1], canvas.shape[0]),
                flags=cv2.INTER_LINEAR,
                borderMode=cv2.BORDER_CONSTANT,
                borderValue=0,
            )
            canvas += warped
        except Exception:
            # 失敗時はフォールバック（外接矩形）
            xs = [p[0] for p in quad_points]
            ys = [p[1] for p in quad_points]
            x0, x1 = int(np.floor(min(xs))), int(np.ceil(max(xs)))
            y0, y1 = int(np.floor(min(ys))), int(np.ceil(max(ys)))
            if x1 <= x0 or y1 <= y0:
                return
            w = max(x1 - x0, 1)
            h = max(y1 - y0, 1)
            gx = np.linspace(-w / 2, w / 2, w)
            gy = np.linspace(-h / 2, h / 2, h)
            gx, gy = np.meshgrid(gx, gy)
            sigma_x = max(w / 5.0, 1.0)
            sigma_y = max(h / 5.0, 1.0)
            g = amplitude * np.exp(-((gx ** 2) / (2 * sigma_x ** 2) + (gy ** 2) / (2 * sigma_y ** 2))).astype(
                np.float32
            )
            x1_clip = min(x0 + w, canvas.shape[1])
            y1_clip = min(y0 + h, canvas.shape[0])
            canvas[y0:y1_clip, x0:x1_clip] += g[: y1_clip - y0, : x1_clip - x0]

    def _draw_quads(self, canvas, quad_list):
        for q in quad_list:
            # q: [x0,y0,x1,y1,x2,y2,x3,y3]
            if len(q) != 8:
                continue
            p = [(q[0],q[1]), (q[2],q[3]), (q[4],q[5]), (q[6],q[7])]
            self._add_perspective_gaussian(canvas, p, amplitude=1.0)

    def generate_gt_heatmaps(
        self,
        gt_info_dic: dict,
        image,                # PIL.Image (リサイズ後)
        original_size=None,   # (orig_w, orig_h)
        ensure_four_channels=True
    ) -> torch.Tensor:
        """
        gt_info_dic 例:
        {
            'main_region':   [[x0,y0,x1,y1,x2,y2,x3,y3], ...],
            'main_affinity': [...],
            'furi_region':   [...],
            'furi_affinity': [...]
        }
        image: 出力ヒートマップと同サイズの PIL.Image (W,H)
        original_size: (orig_w, orig_h) - 元画像サイズ。指定時は座標をスケール。
        return: torch.FloatTensor [C,H,W] (C=4 固定または存在チャネル数)
        """
        w, h = image.size  # PILは (W,H)
        if original_size:
            orig_w, orig_h = original_size
            scale_w = w / orig_w
            scale_h = h / orig_h
        else:
            scale_w = scale_h = 1.0

        channel_names = ['main_region', 'main_affinity', 'furi_region', 'furi_affinity']
        canvases = {name: np.zeros((h, w), dtype=np.float32) for name in channel_names}

        for name in channel_names:
            if name not in gt_info_dic:
                continue
            quads = gt_info_dic[name]
            if not quads:
                continue
            # スケーリング
            arr = np.array(quads, dtype=np.float64)  # (N,8)
            arr[:, ::2] *= scale_w  # x
            arr[:, 1::2] *= scale_h # y
            scaled = arr.tolist()
            self._draw_quads(canvases[name], scaled)

        tensors = []
        for name in channel_names:
            if ensure_four_channels:
                tensors.append(torch.from_numpy(canvases[name]))
            else:
                # 非ゼロ判定で省略したい場合の分岐（今回は常に追加）
                tensors.append(torch.from_numpy(canvases[name]))

        heatmaps = torch.stack(tensors, dim=0)  # [4,H,W]
        return heatmaps

    # 以前の API との後方互換用 (直接利用は推奨しない)
    def _get_gt_tensor(self, path, image, original_size):
        doc_id, image_id = self._get_doc_image_id(Path(path))
        gt_info = self._get_gt_info(doc_id, image_id)
        return self.generate_gt_heatmaps(gt_info_dic=gt_info, image=image, original_size=original_size)

    def get_combine_image(self, image_A, image_B):
        combined_image = (image_A / 2) + (image_B / 2)
        return combined_image

    def show_combine_image(self, image_A, image_B):
        combined_image = self.get_combine_image(image_A, image_B)
        plt.imshow(combined_image, cmap='nipy_spectral')
        plt.title("Combined Image")
        plt.show()

    def show_combine_3chanel_and_1chanel(self, image_A, image_B):
        # Combine a 3-channel image (image_A) and a 1-channel image (image_B)
        # image_A: [3, H, W], image_B: [H, W] or [1, H, W]
        if image_B.ndim == 2:
            image_B = image_B.unsqueeze(0)
        if image_B.shape[0] == 1:
            image_B = image_B.repeat(3, 1, 1)
        combined_image = (image_A + image_B) / 2
        # Convert to numpy for visualization
        combined_image_np = combined_image.permute(1, 2, 0).cpu().numpy()
        plt.imshow(combined_image_np, cmap='nipy_spectral')
        plt.title("Combined 3-channel and 1-channel Image")
        plt.show()
        
    def benchmark_getitem(self, max_samples=None, warmup=0, shuffle=False, verbose=True, dataloader=False, batch_size=4, num_workers=0):
        """__getitem__ または DataLoader 経由の平均時間計測"""
        import random
        dataset_len = len(self)
        if dataset_len == 0:
            raise RuntimeError('Dataset が空です。')
        indices = list(range(dataset_len))
        if shuffle:
            random.shuffle(indices)
        if max_samples is not None:
            indices = indices[:min(max_samples, dataset_len)]

        # ウォームアップ
        for i in range(min(warmup, len(indices))):
            _ = self[indices[i]]

        timings: List[float] = []
        if not dataloader:
            for idx in indices[warmup:]:
                t0 = time.perf_counter()
                _ = self[idx]
                t1 = time.perf_counter()
                timings.append(t1 - t0)
        else:
            loader = torch.utils.data.DataLoader(self, batch_size=batch_size, num_workers=num_workers, shuffle=shuffle)
            taken = 0
            t0 = time.perf_counter()
            for batch in loader:
                taken += len(batch[0])
                if max_samples and taken >= max_samples:
                    break
            t1 = time.perf_counter()
            timings.append(t1 - t0)

        if not timings:
            raise RuntimeError('計測対象がありません。')
        avg = (sum(timings) / len(timings)) / (1 if dataloader else 1)
        if verbose:
            if dataloader:
                print(f'[benchmark_getitem] DataLoader total={timings[0]:.5f}s  batch_size={batch_size} num_workers={num_workers}')
            else:
                total = sum(timings)
                print(f"[benchmark_getitem] samples={len(timings)} avg={avg*1000:.5f} ms total={total:.5f} s min={min(timings)*1000:.5f} ms max={max(timings)*1000:.5f} ms")
        return avg
    