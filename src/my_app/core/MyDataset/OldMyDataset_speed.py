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


class PreTrainDataset_old(Dataset):
    def __init__(self,
                 test_doc_id_list,
                 test_mode=False,
                 # input_path='../kuzushiji_recognition/synthetic_images/input_images/',
                 input_path='../kuzushiji_recognition/synthetic_images/tmp_entire_data/',
                 json_path='../kuzushiji_recognition/synthetic_images/gt_json.json',
                 transform=None,
                 target_width=300,
                 precompute_gt=False,
                 cache_dir=None):  # 事前計算オプションを追加
        super().__init__()
        self.test_doc_id_list = test_doc_id_list
        self.input_path = input_path
        self.transform = transform
        self.target_width = target_width
        self.precompute_gt = precompute_gt
        self.cache_dir = cache_dir or './cache'
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # 画像のIDをリストにして保管。
        self.input_imageID_list = []
        for file_name in os.listdir(self.input_path):
            file_path = os.path.join(self.input_path, file_name)
            if os.path.isfile(file_path) and file_name.split('.')[-1] == 'jpg':
                if not (file_name.split('_sep_')[0] in self.test_doc_id_list) ^ test_mode:
                    self.input_imageID_list.append(file_name.split('.')[0])
        
        # アノテーションデータを保持するjsonファイルをロード
        print(f'before loadGT {len(self.input_imageID_list)}')
        self.gt_json = self.load_GT_json(json_path)
        print(f'after loadGT {len(self.input_imageID_list)}')
        
        # 事前計算を実行
        if self.precompute_gt:
            self.precomputed_gt = self._precompute_ground_truth()
        else:
            self.precomputed_gt = None

    def __len__(self):
        return len(self.input_imageID_list)

    def __getitem__(self, index):
        image_id = self.input_imageID_list[index]
        image = Image.open(self.input_path + image_id + '.jpg')
        
        # 1. 元の画像のサイズを取得
        original_w, original_h = image.size
        
        # 2. アスペクト比を保持したまま、横幅を指定サイズにリサイズ
        aspect_ratio = original_h / original_w
        new_w = self.target_width
        new_h = int(self.target_width * aspect_ratio)
        new_size = (new_h, new_w)
        
        # 3. 画像をリサイズ
        image = functional.resize(image, new_size, interpolation=functional.InterpolationMode.BILINEAR)
        
        # 4. 正解マップを取得（事前計算または実時間生成）
        if self.precomputed_gt and image_id in self.precomputed_gt:
            tensor_gt = self.precomputed_gt[image_id]
        else:
            tensor_gt = self.return_tensor_gt(
                gt_info_dic=self.gt_json['files'][image_id],
                image=image,
                original_size=(original_w, original_h)
            )

        if self.transform:
            image = self.transform(image)
            
        return image, tensor_gt, image_id

    def _precompute_ground_truth(self):
        """正解データを事前計算してキャッシュする"""
        cache_file = os.path.join(self.cache_dir, f'gt_cache_w{self.target_width}.pkl')
        
        # キャッシュディレクトリを作成
        os.makedirs(self.cache_dir, exist_ok=True)
        
        # キャッシュファイルが存在する場合は読み込み
        if os.path.exists(cache_file):
            print(f"正解データのキャッシュを読み込み中: {cache_file}")
            with open(cache_file, 'rb') as f:
                return pickle.load(f)
        
        print("正解データを事前計算中...")
        precomputed_gt = {}
        
        for i, image_id in enumerate(self.input_imageID_list):
            if i % 100 == 0:
                print(f"進捗: {i}/{len(self.input_imageID_list)}")
            
            # 画像サイズを取得
            image = Image.open(self.input_path + image_id + '.jpg')
            original_w, original_h = image.size
            
            # リサイズ後のサイズを計算
            aspect_ratio = original_h / original_w
            new_w = self.target_width
            new_h = int(self.target_width * aspect_ratio)
            resized_image = image.resize((new_w, new_h))
            
            # 正解マップを生成
            tensor_gt = self.return_tensor_gt(
                gt_info_dic=self.gt_json['files'][image_id],
                image=resized_image,
                original_size=(original_w, original_h)
            )
            
            precomputed_gt[image_id] = tensor_gt
        
        # キャッシュに保存
        print(f"正解データをキャッシュに保存中: {cache_file}")
        with open(cache_file, 'wb') as f:
            pickle.dump(precomputed_gt, f)
        
        return precomputed_gt

    def load_GT_json(self, file_path):
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        print("json データを読み込みました。")
        return data

    def return_ground_truth_canvas(self, image):
        w, h = image.size
        main_region = np.zeros((h, w), dtype=np.float32)  # float32に変更
        main_affinity = np.zeros((h, w), dtype=np.float32)
        furi_region = np.zeros((h, w), dtype=np.float32)
        furi_affinity = np.zeros((h, w), dtype=np.float32)
        return main_region, main_affinity, furi_region, furi_affinity

    def create_transformation_grid(self, height, width):
        """GPU上で座標グリッドを作成"""
        y_coords = torch.arange(height, dtype=torch.float32, device=self.device)
        x_coords = torch.arange(width, dtype=torch.float32, device=self.device)
        grid_y, grid_x = torch.meshgrid(y_coords, x_coords, indexing='ij')
        return grid_x, grid_y

    def add_perspective_gaussian_to_canvas_gpu(self, canvas_tensor, points, amplitude=1.0):
        """GPU上でガウシアンマップを生成（高速化版）"""
        # 領域の4点を取得
        src_points = np.array(points, dtype=np.float32)
        
        # サイズ計算
        width = int(max(np.linalg.norm(src_points[0] - src_points[1]), 
                       np.linalg.norm(src_points[2] - src_points[3])))
        height = int(max(np.linalg.norm(src_points[0] - src_points[3]), 
                        np.linalg.norm(src_points[1] - src_points[2])))
        
        width = max(width, 1)
        height = max(height, 1)
        
        # 最小サイズ保証
        min_gaussian_size = 5
        if width < min_gaussian_size or height < min_gaussian_size:
            scale = max(min_gaussian_size / width, min_gaussian_size / height)
            width = max(int(width * scale), min_gaussian_size)
            height = max(int(height * scale), min_gaussian_size)
        
        # GPU上でガウス分布を生成
        device = self.device
        y = torch.linspace(-height / 2, height / 2, height, device=device)
        x = torch.linspace(-width / 2, width / 2, width, device=device)

        grid_y, grid_x = torch.meshgrid(y, x , indexing='ij')
        
        min_sigma = 1.0
        sigma_x = max(width / 5.0, min_sigma)
        sigma_y = max(height / 5.0, min_sigma)
        
        gaussian = amplitude * torch.exp(-((grid_x**2) / (2 * sigma_x**2) + 
                                         (grid_y**2) / (2 * sigma_y**2)))
        
        try:
            # OpenCVでperspective transformation（CPUで実行）
            dst_points = np.array([[0, 0], [width - 1, 0], 
                                 [width - 1, height - 1], [0, height - 1]], dtype=np.float32)
            matrix = cv2.getPerspectiveTransform(dst_points, src_points)
            
            # CPUに移動してOpenCV処理
            gaussian_cpu = gaussian.cpu().numpy().astype(np.float32)
            transformed_gaussian = cv2.warpPerspective(
                gaussian_cpu, matrix, 
                (canvas_tensor.shape[1], canvas_tensor.shape[0]),
                flags=cv2.INTER_LINEAR,
                borderMode=cv2.BORDER_CONSTANT,
                borderValue=0
            )
            
            # GPUに戻す
            transformed_tensor = torch.from_numpy(transformed_gaussian).to(device)
            canvas_tensor += transformed_tensor
            
        except Exception as e:
            print(f"Warning: perspective transformation failed - {e}")
        
        return canvas_tensor

    def add_perspective_gaussian_to_canvas(self, canvas, points, amplitude=1.0):
        """従来のCPU版ガウシアンマップ生成（GPU版から参照されるため追加）"""
        # 領域の4点を取得
        src_points = np.array(points, dtype=np.float32)

        # ガウス分布を生成するための仮想的な正方形領域を定義
        width = int(max(np.linalg.norm(src_points[0] - src_points[1]), np.linalg.norm(src_points[2] - src_points[3])))
        height = int(max(np.linalg.norm(src_points[0] - src_points[3]), np.linalg.norm(src_points[1] - src_points[2])))
        
        # 最小サイズを保証（1ピクセル未満にならないようにする）
        width = max(width, 1)
        height = max(height, 1)
        
        # ガウス分布のサイズを調整（小さすぎる場合は大きくする）
        min_gaussian_size = 5
        if width < min_gaussian_size or height < min_gaussian_size:
            scale = max(min_gaussian_size / width, min_gaussian_size / height)
            width = max(int(width * scale), min_gaussian_size)
            height = max(int(height * scale), min_gaussian_size)
        
        dst_points = np.array([[0, 0], [width - 1, 0], [width - 1, height - 1], [0, height - 1]], dtype=np.float32)

        # ガウス分布を生成
        x = np.linspace(-width / 2, width / 2, width)
        y = np.linspace(-height / 2, height / 2, height)
        x, y = np.meshgrid(x, y)
        
        # シグマ値の調整（小さすぎる場合は最小値を設定）
        min_sigma = 1.0
        sigma_x = max(width / 5.0, min_sigma)
        sigma_y = max(height / 5.0, min_sigma)
        
        gaussian = amplitude * np.exp(-((x**2) / (2 * sigma_x**2) + (y**2) / (2 * sigma_y**2)))

        try:
            # Perspective Transformation行列を計算
            matrix = cv2.getPerspectiveTransform(dst_points, src_points)

            # 入力ガウシアンを適切な形式に変換
            gaussian = gaussian.astype(np.float32)  # float32型に変換
            
            # ガウス分布をPerspective Transformationで変形
            transformed_gaussian = cv2.warpPerspective(
                gaussian,
                matrix,
                (canvas.shape[1], canvas.shape[0]),
                flags=cv2.INTER_LINEAR,
                borderMode=cv2.BORDER_CONSTANT,
                borderValue=0
            )

            # キャンバスにガウス分布を追加
            canvas += transformed_gaussian

        except Exception as e:
            print(f"Warning: perspective transformation failed - {e}")
            # エラーが発生した場合は、キャンバスをそのまま返す

        return canvas

    def design_gaussian_map_gpu(self, canvas, point_list):
        """GPU上でガウシアンマップを生成"""
        canvas_tensor = torch.from_numpy(canvas).to(self.device)
        
        for points in point_list:
            p1x, p1y, p2x, p2y, p3x, p3y, p4x, p4y = points
            canvas_tensor = self.add_perspective_gaussian_to_canvas_gpu(
                canvas_tensor, ((p1x, p1y), (p2x, p2y), (p3x, p3y), (p4x, p4y)), 
                amplitude=1.0
            )
        
        return canvas_tensor.cpu().numpy()

    def design_gaussian_map(self, canvas, point_list):
        # GPU処理が利用可能な場合はGPU版を使用
        if torch.cuda.is_available() and len(point_list) > 5:  # 多くの点がある場合のみGPU使用
            return self.design_gaussian_map_gpu(canvas, point_list)
        
        # 従来のCPU処理
        for points in point_list:
            p1x, p1y, p2x, p2y, p3x, p3y, p4x, p4y = points
            self.add_perspective_gaussian_to_canvas(canvas, ((p1x, p1y), (p2x, p2y), (p3x, p3y), (p4x, p4y)), amplitude=1.0)
        return canvas

    def return_tensor_gt(self, gt_info_dic, image, original_size=None):
        """正解マップを生成する（高速化版）"""
        main_region, main_affinity, furi_region, furi_affinity = self.return_ground_truth_canvas(image)

        # スケーリング係数を計算
        if original_size:
            orig_w, orig_h = original_size
            current_w, current_h = image.size
            scale_w = current_w / orig_w
            scale_h = current_h / orig_h
        else:
            scale_w = scale_h = 1.0

        # キャンバスマップを作成
        canvas_list = []
        canvas_map = {
            'main_region': main_region,
            'main_affinity': main_affinity,
            'furi_region': furi_region,
            'furi_affinity': furi_affinity
        }
        
        for name, canvas in canvas_map.items():
            if name in gt_info_dic:
                # 座標をスケーリング（型キャスト問題を修正）
                points_array = np.array(gt_info_dic[name], dtype=np.float64)  # 明示的にfloat64に変換
                if len(points_array) > 0:
                    # x座標とy座標を分離してスケーリング
                    points_array[:, ::2] *= scale_w  # x座標（偶数インデックス）
                    points_array[:, 1::2] *= scale_h  # y座標（奇数インデックス）
                    scaled_points = points_array.tolist()
                else:
                    scaled_points = []
                
                canvas_list.append(self.design_gaussian_map(canvas, scaled_points))

        # torch tensorに変換（メモリ効率改善）
        tensor_list = []
        for canvas in canvas_list:
            if isinstance(canvas, np.ndarray):
                tensor_list.append(torch.from_numpy(canvas.astype(np.float32)))
            else:
                tensor_list.append(torch.tensor(canvas, dtype=torch.float32))
        
        return_tensor = torch.stack(tensor_list)
        return return_tensor

    @staticmethod
    def create_optimized_dataloader_for_old_dataset(dataset, batch_size=8, num_workers=4):
        """
        PreTrainDataset_old用の最適化されたDataLoaderを作成（高速化版）
        """
        from torch.utils.data import DataLoader
        
        # CPUコア数に基づいてワーカー数を調整
        if num_workers == -1:
            num_workers = min(mp.cpu_count(), 8)
        
        # メモリ効率のための設定
        persistent_workers = num_workers > 0
        prefetch_factor = 4 if persistent_workers else None  # プリフェッチ数を増加
        
        # DataLoaderの作成
        return DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=torch.cuda.is_available(),
            persistent_workers=persistent_workers,
            prefetch_factor=prefetch_factor,
            drop_last=False,
            generator=torch.Generator(),
            # 追加の最適化設定
            timeout=60,  # タイムアウト設定
        )

    def benchmark_dataset(self, num_samples=100):
        """データセットのベンチマークを実行"""
        print(f"ベンチマーク開始: {num_samples}サンプル")
        start_time = time.time()
        
        for i in range(min(num_samples, len(self))):
            if i % 20 == 0:
                print(f"進捗: {i}/{num_samples}")
            _ = self[i]
        
        end_time = time.time()
        total_time = end_time - start_time
        avg_time = total_time / num_samples
        
        print(f"総時間: {total_time:.2f}秒")
        print(f"平均時間/サンプル: {avg_time:.4f}秒")
        print(f"スループット: {num_samples/total_time:.2f}サンプル/秒")