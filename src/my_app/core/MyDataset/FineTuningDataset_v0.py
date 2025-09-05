import os
from pathlib import Path
from typing import List, Optional
from torchvision.transforms import functional
import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from torchvision.transforms import functional as F
import json
import numpy as np
import torch
import cv2
import matplotlib.pyplot as plt


# =============== 設定 ===============
IMAGES_DIR = Path('../../kuzushiji-recognition/char_sep_datas')  # 画像ディレクトリ
GT_JSON_PATH = Path('../../kuzushiji-recognition/char_sep_datas/gt_json.json')    # アノテーションJSON (未使用でもロード例)

# =============== Dataset 最小実装 ===============
class FineTuningDataset_v0(Dataset):
    """最小の自作 Dataset テンプレート

    現段階: 画像をTensor化して (tensor, image_id) を返すのみ。
    後で: JSON 利用 / ターゲット生成 / リサイズ / 前処理 を追加予定。
    """
    def __init__(
            self, 
            images_dir: Path, 
            json_path: Optional[Path] = None, 
            extensions: List[str] = None, 
            transform=None,
            target_width = 300
            ):
        self.images_dir = Path(images_dir)
        self.transform = transform
        self.extensions = extensions or ['.jpg', '.jpeg', '.png']
        self.target_width = target_width
        try:
            import cv2
            self._HAS_CV2 = True
        except ImportError:
            self._HAS_CV2 = False

        assert self.images_dir.exists(), f"画像ディレクトリが存在しません: {self.images_dir}"

        # 画像ファイル列挙
        self.image_paths = self._get_image_path()
        if not self.image_paths:
            raise RuntimeError(f"画像が見つかりません: {self.images_dir}")

        # JSON (必要ならロード) - 今は保持のみ
        self.raw_json = None
        if json_path is not None and json_path.exists():
            with open(json_path, 'r', encoding='utf-8') as f:
                self.raw_json = json.load(f)

    def _get_image_path(self):
        images_path = []
        for doc_path in IMAGES_DIR.iterdir():
            if not doc_path.is_file():
                images_path_obj = doc_path / 'images'
                for image_obj in images_path_obj.iterdir():
                    images_path.append(image_obj)
        return images_path

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx: int):
        path = self.image_paths[idx]
        image = Image.open(path).convert('RGB')
        # 1. 元の画像のサイズを取得
        original_w, original_h = image.size
        
        # 2. アスペクト比を保持したまま、横幅を指定サイズにリサイズ
        aspect_ratio = original_h / original_w
        new_w = self.target_width
        new_h = int(self.target_width * aspect_ratio)
        new_size = (new_h, new_w)
        image = functional.resize(image, new_size, interpolation=functional.InterpolationMode.BILINEAR)

        gt_tensor = self._get_gt_tensor(path, image, original_size=(original_w, original_h))
        if self.transform:
            tensor = self.transform(image)
        else:
            tensor = F.to_tensor(image)  # [0,1] float32
        return tensor, gt_tensor
    
    def _get_doc_image_id(self, path):
        parts = path.parts
        doc_id = parts[-3]
        image_id = parts[-1].split('.')[0]
        return doc_id, image_id
    
    def _get_gt_info(self, doc_id, image_id):
        key = f"{doc_id}_sep_{image_id}"

        if self.raw_json['files'].get(key):
            return self.raw_json['files'][key]
        # データが白紙の場合
        else:
            return {
                'main_region'  : [], 
                'main_affinity': [], 
                'furi_region'  : [], 
                'furi_affinity': []
                }
        

    def _add_perspective_gaussian(self, canvas, quad_points, amplitude=1.0, min_gaussian_size=5):
        """
        quad_points: [(x0,y0),(x1,y1),(x2,y2),(x3,y3)]
        canvas: np.ndarray (H,W) float32
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
            if w < min_gaussian_size or h < min_gaussian_size:
                w = max(w, min_gaussian_size)
                h = max(h, min_gaussian_size)
            gx = np.linspace(-w/2, w/2, w)
            gy = np.linspace(-h/2, h/2, h)
            gx, gy = np.meshgrid(gx, gy)
            sigma_x = max(w/5.0, 1.0)
            sigma_y = max(h/5.0, 1.0)
            g = amplitude * np.exp(-((gx**2)/(2*sigma_x**2) + (gy**2)/(2*sigma_y**2))).astype(np.float32)
            # クリップ範囲
            x1_clip = min(x0 + w, canvas.shape[1])
            y1_clip = min(y0 + h, canvas.shape[0])
            canvas[y0:y1_clip, x0:x1_clip] += g[:y1_clip - y0, :x1_clip - x0]
            return

        src = np.array(quad_points, dtype=np.float32)
        # 推定幅/高さ
        width = int(max(np.linalg.norm(src[0]-src[1]), np.linalg.norm(src[2]-src[3])))
        height = int(max(np.linalg.norm(src[0]-src[3]), np.linalg.norm(src[1]-src[2])))
        width = max(width, 1)
        height = max(height, 1)

        if width < min_gaussian_size or height < min_gaussian_size:
            scale = max(min_gaussian_size/width, min_gaussian_size/height)
            width = max(int(width*scale), min_gaussian_size)
            height = max(int(height*scale), min_gaussian_size)

        dst = np.array([[0,0],[width-1,0],[width-1,height-1],[0,height-1]], dtype=np.float32)

        # ガウス生成
        gx = np.linspace(-width/2, width/2, width)
        gy = np.linspace(-height/2, height/2, height)
        gx, gy = np.meshgrid(gx, gy)
        sigma_x = max(width/5.0, 1.0)
        sigma_y = max(height/5.0, 1.0)
        gaussian = amplitude * np.exp(-((gx**2)/(2*sigma_x**2) + (gy**2)/(2*sigma_y**2))).astype(np.float32)

        try:
            M = cv2.getPerspectiveTransform(dst, src)
            warped = cv2.warpPerspective(
                gaussian, M, (canvas.shape[1], canvas.shape[0]),
                flags=cv2.INTER_LINEAR,
                borderMode=cv2.BORDER_CONSTANT,
                borderValue=0
            )
            canvas += warped
        except Exception as e:
            # 失敗時はフォールバック（外接矩形）
            xs = [p[0] for p in quad_points]
            ys = [p[1] for p in quad_points]
            x0, x1 = int(np.floor(min(xs))), int(np.ceil(max(xs)))
            y0, y1 = int(np.floor(min(ys))), int(np.ceil(max(ys)))
            if x1 <= x0 or y1 <= y0:
                return
            w = max(x1 - x0, 1)
            h = max(y1 - y0, 1)
            gx = np.linspace(-w/2, w/2, w)
            gy = np.linspace(-h/2, h/2, h)
            gx, gy = np.meshgrid(gx, gy)
            sigma_x = max(w/5.0, 1.0)
            sigma_y = max(h/5.0, 1.0)
            g = amplitude * np.exp(-((gx**2)/(2*sigma_x**2) + (gy**2)/(2*sigma_y**2))).astype(np.float32)
            x1_clip = min(x0 + w, canvas.shape[1])
            y1_clip = min(y0 + h, canvas.shape[0])
            canvas[y0:y1_clip, x0:x1_clip] += g[:y1_clip - y0, :x1_clip - x0]

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

    def _get_gt_tensor(self, path, image, original_size):
        doc_id, image_id = self._get_doc_image_id(path)
        gt_info = self._get_gt_info(doc_id, image_id)
        tensor_gt = self.generate_gt_heatmaps(
            gt_info_dic=gt_info, 
            image=image,
            original_size=original_size
        )
        return tensor_gt

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