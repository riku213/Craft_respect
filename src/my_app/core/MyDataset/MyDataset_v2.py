import os
import json
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset
from torchvision.transforms import functional
from PIL import Image
import multiprocessing as mp
import pickle
import time
import hashlib
from functools import lru_cache
from typing import Dict, Any, List, Tuple


class PreTrainDataset_v2(Dataset):
    """
    ベクター(正規化ポリゴン)キャッシュ + 動的レンダリング方式のDataset.
    ポリゴン座標は0-1正規化で保持し、target_width変更時にも再利用。
    ground truthマップは __getitem__ 時に必要サイズへスケールしてガウシアンを射影生成。
    オプションで描画結果もサイズ別diskキャッシュ。
    """
    def __init__(self,
                 test_doc_id_list,
                 test_mode=False,
                 input_path='../kuzushiji_recognition/synthetic_images/tmp_entire_data/',
                 json_path='../kuzushiji_recognition/synthetic_images/gt_json.json',
                 transform=None,
                 target_width=300,
                 precompute_gt=False,  # v2では基本False推奨 (動的生成)
                 cache_dir='./cache',
                 vector_cache_name='vector_cache.pkl',
                 raster_cache_enable=True,
                 raster_cache_max_items=0,  # 0=無制限(注意)。 >0でLRU管理
                 version_tag='v1'):
        super().__init__()
        self.test_doc_id_list = test_doc_id_list
        self.input_path = input_path
        self.transform = transform
        self.target_width = target_width
        self.precompute_gt = precompute_gt  # 従来互換( Trueなら指定widthのラスタを事前生成 )
        self.cache_dir = cache_dir
        self.vector_cache_path = os.path.join(cache_dir, vector_cache_name)
        self.raster_cache_enable = raster_cache_enable
        self.raster_cache_max_items = raster_cache_max_items
        self.version_tag = version_tag
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        os.makedirs(self.cache_dir, exist_ok=True)

        # 画像ID収集
        self.input_imageID_list = []
        for file_name in os.listdir(self.input_path):
            file_path = os.path.join(self.input_path, file_name)
            if os.path.isfile(file_path) and file_name.lower().endswith('.jpg'):
                if not (file_name.split('_sep_')[0] in self.test_doc_id_list) ^ test_mode:
                    self.input_imageID_list.append(os.path.splitext(file_name)[0])

        # JSONロード
        self.gt_json = self._load_gt_json(json_path)

        # ベクターキャッシュ構築/読み込み
        self.vector_cache = self._load_or_build_vector_cache()

        # オプション: 指定 target_width のラスタ事前生成
        self.precomputed_gt = None
        if self.precompute_gt:
            self.precomputed_gt = self._precompute_raster_gt_for_width(self.target_width)

        # メモリ内ラスタLRU (簡易)
        self._raster_mem_cache: Dict[str, torch.Tensor] = {}
        self._raster_mem_cache_access: List[str] = []  # LRU順

    # ========================= 公開メソッド =========================
    def __len__(self):
        return len(self.input_imageID_list)

    def __getitem__(self, index):
        image_id = self.input_imageID_list[index]
        img_path = os.path.join(self.input_path, image_id + '.jpg')
        image = Image.open(img_path).convert('RGB')

        orig_w, orig_h = image.size
        aspect = orig_h / orig_w
        new_w = self.target_width
        new_h = int(round(new_w * aspect))
        image_resized = functional.resize(image, (new_h, new_w), interpolation=functional.InterpolationMode.BILINEAR)

        if self.transform:
            image_tensor = self.transform(image_resized)
        else:
            # ToTensor 相当 (0-1)
            image_tensor = functional.to_tensor(image_resized)

        # GT 取得
        if self.precomputed_gt and image_id in self.precomputed_gt:
            tensor_gt = self.precomputed_gt[image_id]
        else:
            tensor_gt = self._get_raster_gt(image_id, orig_size=(orig_w, orig_h), resized_size=(new_w, new_h))

        return image_tensor, tensor_gt, image_id

    # ========================= ベクターキャッシュ生成 =========================
    def _vector_cache_version_hash(self) -> str:
        meta = {
            'version_tag': self.version_tag,
            'json_items': len(self.gt_json.get('files', {})),
            'image_size_mode': 'from_file_if_exists'  # 画像実寸を使うことで座標ずれ防止
        }
        meta_str = json.dumps(meta, sort_keys=True)
        return hashlib.md5(meta_str.encode('utf-8')).hexdigest()

    def _load_gt_json(self, path):
        with open(path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        return data

    def _normalize_polygon_points(self, pts: List[float], orig_w: int, orig_h: int) -> List[float]:
        # pts = [x1,y1,x2,y2,...]
        normed = []
        for i, v in enumerate(pts):
            if i % 2 == 0:  # x
                normed.append(float(v) / orig_w if orig_w > 0 else 0.0)
            else:          # y
                normed.append(float(v) / orig_h if orig_h > 0 else 0.0)
        return normed

    def _build_vector_entry(self, file_id: str, file_entry: Dict[str, Any]) -> Dict[str, Any]:
        # original_size を JSON or 画像ファイルから取得
        orig_w, orig_h = None, None
        if 'original_size' in file_entry and isinstance(file_entry['original_size'], (list, tuple)) and len(file_entry['original_size']) >= 2:
            ow, oh = file_entry['original_size'][:2]
            if isinstance(ow, (int, float)) and isinstance(oh, (int, float)) and ow and oh:
                orig_w, orig_h = int(ow), int(oh)

        if orig_w is None or orig_h is None:
            img_path = os.path.join(self.input_path, file_id + '.jpg')
            if os.path.exists(img_path):
                try:
                    with Image.open(img_path) as im:
                        orig_w, orig_h = im.size  # PILは (w,h)
                except Exception:
                    orig_w = orig_h = None

        # それでも無ければポリゴン最大値から推測
        if orig_w is None or orig_h is None:
            max_x = 0
            max_y = 0
            for key in ['main_region', 'main_affinity', 'furi_region', 'furi_affinity']:
                for poly in file_entry.get(key, []):
                    xs = poly[0::2]
                    ys = poly[1::2]
                    if xs: max_x = max(max_x, max(xs))
                    if ys: max_y = max(max_y, max(ys))
            orig_w = int(max_x) + 1 if max_x else 1
            orig_h = int(max_y) + 1 if max_y else 1

        orig_w = max(int(orig_w), 1)
        orig_h = max(int(orig_h), 1)

        norm_data = {}
        for key in ['main_region', 'main_affinity', 'furi_region', 'furi_affinity']:
            polys = file_entry.get(key, []) or []
            norm_data[key] = [self._normalize_polygon_points(poly, orig_w, orig_h) for poly in polys]

        return {
            'original_size': [orig_w, orig_h],
            'normalized': norm_data
        }

    def _load_or_build_vector_cache(self):
        version_hash = self._vector_cache_version_hash()
        if os.path.exists(self.vector_cache_path):
            try:
                with open(self.vector_cache_path, 'rb') as f:
                    data = pickle.load(f)
                if data.get('_version_hash') == version_hash:
                    return data['entries']
            except Exception:
                pass
        # 再構築
        entries = {}
        files_dic = self.gt_json.get('files', {})
        for file_id, file_entry in files_dic.items():
            try:
                entries[file_id] = self._build_vector_entry(file_id, file_entry)
            except Exception as e:
                print(f"[vector_cache] skip {file_id}: {e}")
        payload = {'_version_hash': version_hash, 'entries': entries}
        with open(self.vector_cache_path, 'wb') as f:
            pickle.dump(payload, f)
        return entries

    # ========================= ラスタ生成 =========================
    def _get_raster_cache_key(self, image_id: str, resized_size: Tuple[int, int]) -> str:
        w, h = resized_size
        return f"{image_id}_w{w}_h{h}"

    def _touch_lru(self, key: str):
        if self.raster_cache_max_items <= 0:
            return
        if key in self._raster_mem_cache_access:
            self._raster_mem_cache_access.remove(key)
        self._raster_mem_cache_access.append(key)
        # eviction
        while len(self._raster_mem_cache_access) > self.raster_cache_max_items:
            evict = self._raster_mem_cache_access.pop(0)
            self._raster_mem_cache.pop(evict, None)

    def _get_raster_gt(self, image_id: str, orig_size: Tuple[int, int], resized_size: Tuple[int, int]) -> torch.Tensor:
        key = self._get_raster_cache_key(image_id, resized_size)
        if self.raster_cache_enable and key in self._raster_mem_cache:
            self._touch_lru(key)
            return self._raster_mem_cache[key]

        vector_entry = self.vector_cache.get(image_id)
        if not vector_entry:
            # 空テンソル(4ch)返す
            w, h = resized_size
            empty = torch.zeros(4, h, w, dtype=torch.float32)
            return empty

        orig_w, orig_h = vector_entry['original_size']
        target_w, target_h = resized_size
        scale_w = target_w / orig_w
        scale_h = target_h / orig_h

        channel_order = ['main_region', 'main_affinity', 'furi_region', 'furi_affinity']
        canvases = []
        for cname in channel_order:
            polys = vector_entry['normalized'][cname]
            canvas = np.zeros((target_h, target_w), dtype=np.float32)
            if polys:
                scaled_polys = []
                for poly in polys:
                    # poly: normalized [x1,y1,...]
                    pts = []
                    for i, v in enumerate(poly):
                        if i % 2 == 0:
                            pts.append(v * target_w)
                        else:
                            pts.append(v * target_h)
                    scaled_polys.append(pts)
                canvas = self._design_gaussian_map_cpu(canvas, scaled_polys)
            canvases.append(torch.from_numpy(canvas))

        tensor_gt = torch.stack(canvases, dim=0)

        if self.raster_cache_enable:
            self._raster_mem_cache[key] = tensor_gt
            self._touch_lru(key)
        return tensor_gt

    # ========================= CPU Gaussian (バッチ高速化簡易) =========================
    def _design_gaussian_map_cpu(self, canvas: np.ndarray, point_list: List[List[float]]) -> np.ndarray:
        for points in point_list:
            p1x, p1y, p2x, p2y, p3x, p3y, p4x, p4y = points
            self._add_perspective_gaussian_cpu(canvas, ((p1x, p1y), (p2x, p2y), (p3x, p3y), (p4x, p4y)))
        return canvas

    def _add_perspective_gaussian_cpu(self, canvas: np.ndarray, pts, amplitude=1.0):
        src_points = np.array(pts, dtype=np.float32)
        width = int(max(np.linalg.norm(src_points[0] - src_points[1]), np.linalg.norm(src_points[2] - src_points[3])))
        height = int(max(np.linalg.norm(src_points[0] - src_points[3]), np.linalg.norm(src_points[1] - src_points[2])))
        width = max(width, 1)
        height = max(height, 1)
        min_gaussian_size = 5
        if width < min_gaussian_size or height < min_gaussian_size:
            scale = max(min_gaussian_size / width, min_gaussian_size / height)
            width = max(int(width * scale), min_gaussian_size)
            height = max(int(height * scale), min_gaussian_size)

        dst_points = np.array([[0,0],[width-1,0],[width-1,height-1],[0,height-1]], dtype=np.float32)
        x = np.linspace(-width/2, width/2, width)
        y = np.linspace(-height/2, height/2, height)
        xg, yg = np.meshgrid(x, y)
        min_sigma = 1.0
        sigma_x = max(width/5.0, min_sigma)
        sigma_y = max(height/5.0, min_sigma)
        gaussian = amplitude * np.exp(-((xg**2)/(2*sigma_x**2) + (yg**2)/(2*sigma_y**2)))
        try:
            matrix = cv2.getPerspectiveTransform(dst_points, src_points)
            gaussian = gaussian.astype(np.float32)
            transformed = cv2.warpPerspective(
                gaussian, matrix, (canvas.shape[1], canvas.shape[0]),
                flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=0
            )
            canvas += transformed
        except Exception:
            pass
        return canvas

    # ========================= 旧式互換: 事前ラスタ生成 =========================
    def _precompute_raster_gt_for_width(self, width: int):
        cache_file = os.path.join(self.cache_dir, f'pre_raster_w{width}.pkl')
        if os.path.exists(cache_file):
            try:
                with open(cache_file, 'rb') as f:
                    return pickle.load(f)
            except Exception:
                pass
        pre = {}
        for image_id in self.input_imageID_list:
            img_path = os.path.join(self.input_path, image_id + '.jpg')
            try:
                with Image.open(img_path) as im:
                    orig_w, orig_h = im.size
                aspect = orig_h / orig_w
                target_h = int(round(width * aspect))
                pre[image_id] = self._get_raster_gt(image_id, (orig_w, orig_h), (width, target_h))
            except Exception as e:
                print(f"[precompute] skip {image_id}: {e}")
        with open(cache_file, 'wb') as f:
            pickle.dump(pre, f)
        return pre

    # ========================= ベンチマーク =========================
    def benchmark_dataset(self, num_samples=100):
        print(f"benchmark start: {num_samples}")
        st = time.time()
        for i in range(min(num_samples, len(self))):
            _ = self[i]
        et = time.time()
        total = et - st
        print(f"total {total:.2f}s  avg {total/num_samples:.4f}s  throughput {num_samples/total:.2f}/s")