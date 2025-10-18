import sys
from pathlib import Path
ROOT = Path.cwd()
sys.path.insert(0, str(ROOT))
import os
from pathlib import Path
import sys
from matplotlib import pyplot as plt
from src.my_app.core.MyDataset.FineTuningDataset_v1 import FineTuningDataset_v1
from src.my_app.core.MyDataset import FineTuningDataset_v0
from torch.utils.data import DataLoader
from torch.utils.data import DataLoader
from src.my_app import UNet, create_optimized_dataloader
import torch
from tqdm import tqdm
from src.my_app.utils.cood_manager import CoodManager
import torch.nn.functional as F


class PredictImages:
    def __init__(self,
                IMAGES_DIR = '../../kuzushiji-recognition/char_sep_datas',  # 画像ディレクトリ
                GT_JSON_PATH = '../../kuzushiji-recognition/char_sep_datas/gt_json.json',    # アノテーションJSON (未使用でもロード例)        
                checkpoint_dir = "../.checkpoints",
                checkpoint_dir_finetuning = "../checkpoints_finetuning"
):
        test_doc_id_list = [
            '200021637',
            '100249371',
            '100249537',
            '200005598',
            '200014740',
            '200020019',
            '200021712',
            '200021869'
        ]
        self.IMAGES_DIR = IMAGES_DIR
        self.GT_JSON_PATH = GT_JSON_PATH
        self.train_dataset = FineTuningDataset_v1(
            test_doc_id=test_doc_id_list,
            test_mode=False,
            images_dir=Path(IMAGES_DIR), 
            json_path=Path(GT_JSON_PATH),
            precompute_gt=True,
            target_width=300,
            )
        self.test_dataset = FineTuningDataset_v1(
            test_doc_id=test_doc_id_list,
            test_mode=True,
            images_dir=Path(IMAGES_DIR),
            json_path=Path(GT_JSON_PATH),
            precompute_gt=True,
            target_width=300,
        )
        self.train_loader = DataLoader(self.train_dataset, batch_size=1, shuffle=False, num_workers=4, pin_memory=True)
        self.test_loader = DataLoader(self.test_dataset, batch_size=1, shuffle=False, num_workers=4, pin_memory=True)

        self.check_point_version = '1.1'
        self.checkpoint_dir = checkpoint_dir
        self.checkpoint_dir_finetuning = checkpoint_dir_finetuning
        self.checkpoint_path = os.path.join(self.checkpoint_dir, f"latest_checkpoint_V{self.check_point_version}.pth")

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = UNet(3, 4).to(self.device)
        
        self.criterion = self.weighted_mse_loss # 回帰問題なのでMSE損失を使用 
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=1e-3)
        os.makedirs(self.checkpoint_dir, exist_ok=True)

        # 最良のモデルを追跡するための変数
        self.best_test_loss = float('inf')
        self.start_epoch = 0

        # 損失の履歴を保存するリストを初期化
        self.train_loss_history = []
        self.test_loss_history = []

        # チェックポイントの読み込み（存在する場合）
        if os.path.exists(self.checkpoint_path):
            self.checkpoint = torch.load(self.checkpoint_path, map_location='cpu')
            self.model.load_state_dict(self.checkpoint['model_state_dict'])
            # optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            self.start_epoch = 0
            self.best_test_loss = self.checkpoint.get('best_test_loss', float('inf'))
            self.train_loss_history = self.checkpoint.get('train_loss_history', self.train_loss_history)
            self.test_loss_history  = self.checkpoint.get('test_loss_history',  self.test_loss_history)
            # print(f"[PredictImages]: チェックポイントを読み込みました（エポック {self.start_epoch}）")
        else:
            self.start_epoch = 0
            self.best_test_loss = float('inf')
        # 以降は train_loss_history / test_loss_history を再初期化しない

    def weighted_mse_loss(self, pred, target, thresh=0.01, pos_weight=20, use_continuous=True, eps=1e-8):
        """
        pred: (B,C,H,W)
        target: (B,C,H,W)
        thresh: 正例二値化の閾値 (use_continuous=False のとき)
        pos_weight: 正例に掛ける重み (または連続重み係数 α)
        use_continuous: Trueなら w = 1 + pos_weight * target（ターゲット値で連続的に重み付け）
        """
        if use_continuous:
            w = 1.0 + pos_weight * target   # target が 0～1 と想定
        else:
            pos = (target > thresh).float()
            w = 1.0 + (pos_weight - 1.0) * pos
        diff2 = (pred - target) ** 2
        loss = (w * diff2).sum() / (w.sum().clamp_min(eps))
        return loss

    def crop_labels_to_match(self, labels_to_crop, target_tensor):
        target_h, target_w = target_tensor.shape[2:]
        source_h, source_w = labels_to_crop.shape[2:]
        delta_h = (source_h - target_h) // 2
        delta_w = (source_w - target_w) // 2
        return labels_to_crop[:, :, delta_h:delta_h + target_h, delta_w:delta_w + target_w]
    
    def resize_keep_aspect_to_width(self, imgs: torch.Tensor, target_w: int):
        """
        imgs: [N,C,H,W] または [C,H,W] の Torch テンソル（float推奨, 0..1）
        target_w: 目的の横幅（ピクセル）
        戻り値: resized_imgs([N,C,new_h,target_w] or [C,new_h,target_w]), (sy, sx)
        """
        single = (imgs.ndim == 3)
        if single:
            imgs = imgs.unsqueeze(0)  # [1,C,H,W]

        N, C, H, W = imgs.shape
        if W == 0:
            raise ValueError("Width is zero.")
        new_h = max(1, int(round(H * (target_w / W))))
        size = (new_h, target_w)

        # antialias は PyTorch>=2.0 で有効。古い版でも動くように try/except。
        try:
            resized = F.interpolate(imgs, size=size, mode="bilinear", align_corners=False, antialias=True)
        except TypeError:
            resized = F.interpolate(imgs, size=size, mode="bilinear", align_corners=False)

        sx = target_w / W
        sy = new_h / H

        if single:
            resized = resized.squeeze(0)  # [C,new_h,target_w]
        return resized, (sy, sx)
    def predict(self, test_mode=True):
        epoch = 0
        num_epochs = 1

        checkpoint_dir = Path(self.checkpoint_dir_finetuning)
        check_point_version = '1.1'
        checkpoint_path = os.path.join(checkpoint_dir, f"latest_checkpoint_V{check_point_version}.pth")

        # チェックポイントの読み込み（存在する場合）
        if os.path.exists(checkpoint_path):
            checkpoint = torch.load(checkpoint_path, map_location='cpu')
            self.model.load_state_dict(checkpoint['model_state_dict'])
            # optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            start_epoch = 0
            best_test_loss = checkpoint.get('best_test_loss', float('inf'))
            self.train_loss_history = checkpoint.get('train_loss_history', self.train_loss_history)
            self.test_loss_history  = checkpoint.get('test_loss_history',  self.test_loss_history)
            print(f"[PredictImages]: チェックポイントを読み込みました（エポック {start_epoch}）")
        else:
            start_epoch = 0
            best_test_loss = float('inf')
            print(f"[PredictImages]: チェックポイントが見つかりません。{checkpoint_path}新しいモデルで開始します。")
        # 以降は train_loss_history / test_loss_history を再初期化しない

        cood_manager = CoodManager(csv_path_dir=self.IMAGES_DIR)
        if test_mode:
            cood_list = cood_manager.get_cood_list(self.test_dataset.get_image_paths())
            orig_size_list = self.test_dataset.get_orig_size_list()
            test_bar = tqdm(self.test_loader, desc=f"[PredictImages] Epoch {epoch+1}/{num_epochs} [Train]")
        else:
            cood_list = cood_manager.get_cood_list(self.train_dataset.get_image_paths())
            orig_size_list = self.train_dataset.get_orig_size_list()
            test_bar = tqdm(self.train_loader, desc=f"[PredictImages] Epoch {epoch+1}/{num_epochs} [Train]")

        image_list = []
        teacher_heatmap_list = []
        pred_heatmap_list = []
        pred_heatmap_list_100 = []
        pred_heatmap_list_200 = []
        for batch in test_bar:
            # バッチは (imgs, masks, file_id) 想定
            if len(batch) == 3:
                imgs = batch[0]
                masks = batch[1]
                file_ids = batch[2]
            elif len(batch) == 2:
                imgs = batch[0]
                resized_imgs100, (sy, sx) = self.resize_keep_aspect_to_width(imgs, target_w=100)
                resized_imgs200, (sy, sx) = self.resize_keep_aspect_to_width(imgs, target_w=200)
                masks = batch[1]
                file_ids = None
            pred = self.model(imgs.to(self.device))
            pred_100 = self.model(resized_imgs100.to(self.device))
            pred_200 = self.model(resized_imgs200.to(self.device))
            cropped_imgs = self.crop_labels_to_match(imgs, pred)
            cropped_masks = self.crop_labels_to_match(masks, pred)
            image_list.append(cropped_imgs.cpu().detach()[0])
            teacher_heatmap_list.append(cropped_masks.cpu().detach()[0])
            pred_heatmap_list.append(pred.cpu().detach()[0])
            pred_heatmap_list_100.append(pred_100.cpu().detach()[0])
            pred_heatmap_list_200.append(pred_200.cpu().detach()[0])
        return image_list, teacher_heatmap_list, pred_heatmap_list , cood_list, orig_size_list, pred_heatmap_list_100, pred_heatmap_list_200

    def convert_boxes_to_orgsize(boxes, box_width, org_width, org_height):
        box_height = box_width * (org_height / org_width)

        rate_hor = org_width / box_width
        rate_ver = org_height / box_height

        resized_boxes = []
        for box in boxes:
            x, y, w, h = box
            rx = (x * rate_hor)
            ry = (y * rate_ver)
            rw = (w * rate_hor)
            rh = (h * rate_ver)
            resized_boxes.append([rx, ry, rw, rh])
            