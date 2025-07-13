from torch.utils.data import Dataset
import torch
import torch.nn.functional as F
import os
import json
from PIL import Image
import numpy as np
import cv2
import torchvision.transforms.functional as functional
from typing import List, Tuple
import multiprocessing as mp
from functools import lru_cache

class PreTrainDataset(Dataset):
    def __init__(self,
                 test_doc_id_list,
                 test_mode = False,
                 input_path = '../kuzushiji-recognition/synthetic_images/input_images/',
                 json_path = '../kuzushiji-recognition/synthetic_images/gt_json.json',
                 transform = None,
                 image_downsample_rate = 10,
                 device = None,
                 precompute_gt = True,
                 num_workers = None):
        super().__init__()
        self.test_doc_id_list = test_doc_id_list
        self.input_path = input_path
        self.transform = transform
        self.image_downsample_rate = image_downsample_rate
        
        # デバイス設定
        if device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = device
            
        print(f"Using device: {self.device}")
        
        # ワーカー数設定
        if num_workers is None:
            self.num_workers = min(mp.cpu_count(), 4)
        else:
            self.num_workers = num_workers
        
        # 画像のIDをリストにして保管
        self.input_imageID_list = []
        for file_name in os.listdir(self.input_path):
            file_path = os.path.join(self.input_path, file_name)
            if os.path.isfile(file_path):
                if not (file_name.split('_sep_')[0] in self.test_doc_id_list) ^ test_mode:
                    self.input_imageID_list.append(file_name.split('.')[0])
        
        # アノテーションデータを保持するjsonファイルをロード
        self.gt_json = self.load_GT_json(json_path)
        
        # 正解データの事前計算（オプション）
        self.precomputed_gt = {}
        if precompute_gt:
            print("Pre-computing ground truth data...")
            self._precompute_ground_truth()
            print("Pre-computation completed.")

    def __len__(self):
        return len(self.input_imageID_list)
    
    def __getitem__(self, index):
        image_id = self.input_imageID_list[index]
        image = Image.open(self.input_path + image_id + '.jpg')
        
        # 事前計算されたデータがあれば使用
        if image_id in self.precomputed_gt:
            tensor_gt = self.precomputed_gt[image_id]
        else:
            # リアルタイムで正解データを生成
            tensor_gt = self.return_tensor_gt_optimized(
                gt_info_dic=self.gt_json['files'][image_id], 
                image=image
            )
        
        # 1. 元の画像のサイズを取得
        original_w, original_h = image.size
        # 2. ターゲットとなる新しいサイズを計算
        new_size = (original_h // self.image_downsample_rate, original_w // self.image_downsample_rate)
        
        # 3. リサイズ処理
        image = functional.resize(image, new_size, interpolation=functional.InterpolationMode.BILINEAR)
        tensor_gt = F.interpolate(
            tensor_gt.unsqueeze(0), 
            size=new_size, 
            mode='bilinear', 
            align_corners=False
        ).squeeze(0)

        if self.transform:
            image = self.transform(image)
        
        return image, tensor_gt

    def _precompute_ground_truth(self):
        """正解データを事前計算してメモリに保存"""
        for image_id in self.input_imageID_list: #check
            try:
                image = Image.open(self.input_path + image_id + '.jpg')
                tensor_gt = self.return_tensor_gt_optimized(
                    gt_info_dic=self.gt_json['files'][image_id], 
                    image=image
                )
                self.precomputed_gt[image_id] = tensor_gt
            except Exception as e:
                print(f"Error precomputing {image_id}: {e}")
                continue

    def load_GT_json(self, file_path):
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        print("jsonデータを読み込みました。")
        return data

    def return_tensor_gt_optimized(self, gt_info_dic, image):
        """最適化された正解データ生成メソッド"""
        w, h = image.size
        
        # GPUでテンソルを直接作成
        canvas_tensors = torch.zeros(4, h, w, dtype=torch.float32, device=self.device)
        
        # 各チャネルを並列処理
        channel_names = ['main_region', 'main_affinity', 'furi_region', 'furi_affinity']
        
        for i, channel_name in enumerate(channel_names):
            if channel_name in gt_info_dic and gt_info_dic[channel_name]:
                canvas_tensors[i] = self.design_gaussian_map_gpu(
                    canvas_tensors[i], 
                    gt_info_dic[channel_name], 
                    w, h
                )
        
        return canvas_tensors

    def design_gaussian_map_gpu(self, canvas_tensor, point_list, width, height):
        """GPU上でガウス分布マップを生成"""
        if not point_list:
            return canvas_tensor
            
        # バッチ処理のためにポイントリストを整理
        batch_points = torch.tensor(point_list, dtype=torch.float32, device=self.device)
        
        for points in batch_points:
            p1x, p1y, p2x, p2y, p3x, p3y, p4x, p4y = points
            
            # 四角形の各頂点
            src_points = torch.tensor([
                [p1x, p1y], [p2x, p2y], [p3x, p3y], [p4x, p4y]
            ], dtype=torch.float32, device=self.device)
            
            canvas_tensor = self.add_perspective_gaussian_gpu(
                canvas_tensor, src_points, width, height
            )
        
        return canvas_tensor

    def add_perspective_gaussian_gpu(self, canvas, src_points, canvas_width, canvas_height):
        """GPU上で透視変換されたガウス分布を追加"""
        # 四角形のサイズを計算
        width = max(
            torch.norm(src_points[0] - src_points[1]).item(),
            torch.norm(src_points[2] - src_points[3]).item()
        )
        height = max(
            torch.norm(src_points[0] - src_points[3]).item(),
            torch.norm(src_points[1] - src_points[2]).item()
        )
        
        width = int(width) + 1
        height = int(height) + 1
        
        # ガウス分布を生成
        gaussian = self.create_gaussian_kernel_gpu(width, height)
        
        # 透視変換行列を計算（CPUで実行）
        src_np = src_points.cpu().numpy()
        dst_np = np.array([
            [0, 0], [width-1, 0], [width-1, height-1], [0, height-1]
        ], dtype=np.float32)
        
        try:
            matrix = cv2.getPerspectiveTransform(dst_np, src_np)
            
            # 変換をGPU上で実行
            transformed_gaussian = self.warp_perspective_gpu(
                gaussian, matrix, canvas_width, canvas_height
            )
            
            canvas += transformed_gaussian
            
        except cv2.error:
            # 透視変換が失敗した場合はスキップ
            pass
            
        return canvas

    @lru_cache(maxsize=128)
    def create_gaussian_kernel_gpu(self, width, height):
        """GPU上でガウシアンカーネルを生成（キャッシュ付き）"""
        x = torch.linspace(-width/2, width/2, width, device=self.device)
        y = torch.linspace(-height/2, height/2, height, device=self.device)
        
        # メッシュグリッドを作成
        y_grid, x_grid = torch.meshgrid(y, x, indexing='ij')
        
        # ガウス分布のパラメータ
        sigma_x = width / 5.0
        sigma_y = height / 5.0
        
        # ガウス分布を計算
        gaussian = torch.exp(-(x_grid**2 / (2 * sigma_x**2) + y_grid**2 / (2 * sigma_y**2)))
        
        return gaussian

    def warp_perspective_gpu(self, image_tensor, matrix, output_width, output_height):
        """GPU上で透視変換を実行"""
        # 変換行列をテンソルに変換
        matrix_tensor = torch.from_numpy(matrix).float().to(self.device)
        
        # グリッドを生成
        grid = self.create_transformation_grid(
            matrix_tensor, output_height, output_width
        )
        
        # grid_sampleを使用して変換
        image_batch = image_tensor.unsqueeze(0).unsqueeze(0)  # [1, 1, H, W]
        grid_batch = grid.unsqueeze(0)  # [1, H, W, 2]
        
        transformed = F.grid_sample(
            image_batch, grid_batch, 
            mode='bilinear', 
            padding_mode='zeros',
            align_corners=False
        )
        
        return transformed.squeeze(0).squeeze(0)

    def create_transformation_grid(self, matrix, height, width):
        """変換グリッドを作成"""
        # 出力座標を生成
        y_coords = torch.arange(height, dtype=torch.float32, device=self.device)
        x_coords = torch.arange(width, dtype=torch.float32, device=self.device)
        
        y_grid, x_grid = torch.meshgrid(y_coords, x_coords, indexing='ij')
        
        # 同次座標に変換
        ones = torch.ones_like(x_grid)
        coords = torch.stack([x_grid, y_grid, ones], dim=-1)  # [H, W, 3]
        
        # 逆変換行列を適用
        try:
            inv_matrix = torch.inverse(matrix)
        except:
            # 逆行列が計算できない場合は単位行列を使用
            inv_matrix = torch.eye(3, device=self.device)
        
        # 変換を適用
        transformed_coords = torch.matmul(coords, inv_matrix.T)  # [H, W, 3]
        
        # 正規化座標に変換
        x_norm = transformed_coords[..., 0] / transformed_coords[..., 2]
        y_norm = transformed_coords[..., 1] / transformed_coords[..., 2]
        
        # grid_sampleの座標系に変換 [-1, 1]
        grid_x = 2.0 * x_norm / (width - 1) - 1.0
        grid_y = 2.0 * y_norm / (height - 1) - 1.0
        
        grid = torch.stack([grid_x, grid_y], dim=-1)
        
        return grid


# 使用例とパフォーマンス比較
def benchmark_dataset(dataset, num_samples=10):
    """データセットのパフォーマンスをベンチマーク"""
    import time
    
    start_time = time.time()
    for i in range(min(num_samples, len(dataset))):
        image, gt = dataset[i]
    end_time = time.time()
    
    avg_time = (end_time - start_time) / num_samples
    print(f"Average time per sample: {avg_time:.4f} seconds")
    return avg_time


# データローダー用の高速化設定
def create_optimized_dataloader(dataset, batch_size=8, num_workers=4):
    """最適化されたDataLoaderを作成"""
    from torch.utils.data import DataLoader
    
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True if torch.cuda.is_available() else False,
        persistent_workers=True if num_workers > 0 else False,
        prefetch_factor=2 if num_workers > 0 else 2
    )