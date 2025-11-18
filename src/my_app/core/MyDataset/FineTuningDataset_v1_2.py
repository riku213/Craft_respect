from __future__ import annotations
from typing import Any, Tuple
import numpy as np
from PIL import Image

# パッケージ/スクリプト両対応の相対/絶対インポート
try:
    from .FineTuningDataset_v1 import FineTuningDataset_v1
except ImportError:
    from FineTuningDataset_v1 import FineTuningDataset_v1


__all__ = ["FineTuningDataset_v1_2"]


class FineTuningDataset_v1_2(FineTuningDataset_v1):
    """
    FineTuningDataset_v1 を継承した準備版。
    - 現状は v1 と同一挙動
    - 将来の拡張用フック (_before_getitem / _after_getitem) を用意
    - 追加オプションの例: enable_new_logic（現時点では未使用）
    """

    def __init__(self, *args: Any, enable_new_logic: bool = False, **kwargs: Any) -> None:
        self.enable_new_logic = enable_new_logic  # 将来の切替スイッチ用
        # ガウスのテンプレート（1回だけ生成して使い回す）
        self._gaussian_template: np.ndarray | None = None
        self._gaussian_base_size: int = 256  # ベース解像度（必要に応じて調整可）
        super().__init__(*args, **kwargs)
        print('[debug]: FineTuningDataset_v1_2 initialized.')

    def _get_gaussian_template(self) -> np.ndarray:
        """
        一度だけ生成し、以降は使い回すベースガウスマップ（float32, [H,W]）。
        sigma は base_size/5 に設定（v1 と同等の見た目を目安）。
        """
        if self._gaussian_template is None:
            bw = bh = int(self._gaussian_base_size)
            gx = np.linspace(-bw/2, bw/2, bw, dtype=np.float32)
            gy = np.linspace(-bh/2, bh/2, bh, dtype=np.float32)
            gx, gy = np.meshgrid(gx, gy)
            sigma_x = max(bw / 5.0, 1.0)
            sigma_y = max(bh / 5.0, 1.0)
            g = np.exp(-((gx**2) / (2 * sigma_x**2) + (gy**2) / (2 * sigma_y**2))).astype(np.float32)
            self._gaussian_template = g
        return self._gaussian_template

    def _add_perspective_gaussian(self, canvas, quad_points, amplitude=1.0, min_gaussian_size=5):
        """
        双線形写像でガウスマップを台形へ変形しつつピークを中心に保持。
        さらに、forward 散布時の量子化アーチファクトを抑えるため双一次スプラットで加算する。
        quad_points: [(x0,y0),(x1,y1),(x2,y2),(x3,y3)]（時計回り想定）
        """
        quad = np.asarray(quad_points, dtype=np.float32)
        if quad.shape != (4, 2):
            return

        # 面積チェック（ほぼ線ならスキップ）
        area = 0.5 * abs(
            quad[0,0]*quad[1,1] + quad[1,0]*quad[2,1] + quad[2,0]*quad[3,1] + quad[3,0]*quad[0,1]
            - quad[1,0]*quad[0,1] - quad[2,0]*quad[1,1] - quad[3,0]*quad[2,1] - quad[0,0]*quad[3,1]
        )
        if area < 1e-3:
            return

        # 目標解像度（辺長ベース）
        edge_w = int(max(np.linalg.norm(quad[0]-quad[1]), np.linalg.norm(quad[2]-quad[3])))
        edge_h = int(max(np.linalg.norm(quad[0]-quad[3]), np.linalg.norm(quad[1]-quad[2])))
        grid_w = max(edge_w, min_gaussian_size)
        grid_h = max(edge_h, min_gaussian_size)

        # テンプレートを取得して必要ならリサイズ（1度生成したものを使い回し）
        base = self._get_gaussian_template()
        if base.shape != (grid_h, grid_w):
            g_img = Image.fromarray(base)
            gaussian = np.array(g_img.resize((grid_w, grid_h), resample=Image.BILINEAR), dtype=np.float32)
        else:
            gaussian = base
        if amplitude != 1.0:
            gaussian = gaussian * float(amplitude)

        # 正規化グリッド（u:横, v:縦）
        u = np.linspace(0.0, 1.0, grid_w, dtype=np.float32)
        v = np.linspace(0.0, 1.0, grid_h, dtype=np.float32)
        uu, vv = np.meshgrid(u, v)

        p0, p1, p2, p3 = quad
        one_minus_u = 1.0 - uu
        one_minus_v = 1.0 - vv

        # 双線形写像（中心は u=v=0.5 で台形の双線形中心に写る）
        x = (one_minus_u * one_minus_v) * p0[0] + (uu * one_minus_v) * p1[0] + (uu * vv) * p2[0] + (one_minus_u * vv) * p3[0]
        y = (one_minus_u * one_minus_v) * p0[1] + (uu * one_minus_v) * p1[1] + (uu * vv) * p2[1] + (one_minus_u * vv) * p3[1]

        Hc, Wc = canvas.shape

        # 双一次スプラット：4 近傍へ重み分配して加算（最近傍散布の縦横の筋を解消）
        x0 = np.floor(x).astype(np.int32)
        y0 = np.floor(y).astype(np.int32)
        fx = x - x0
        fy = y - y0

        # キャンバス内で +1 がはみ出さない領域に限定
        mask = (x0 >= 0) & (x0 < Wc - 1) & (y0 >= 0) & (y0 < Hc - 1)
        if not np.any(mask):
            return

        x0 = x0[mask]; y0 = y0[mask]
        fx = fx[mask]; fy = fy[mask]
        gv = gaussian[mask]

        w00 = (1.0 - fx) * (1.0 - fy)
        w10 = fx * (1.0 - fy)
        w01 = (1.0 - fx) * fy
        w11 = fx * fy

        # フラット化して安全に散布
        x1 = x0 + 1
        y1 = y0 + 1

        np.add.at(canvas, (y0.ravel(), x0.ravel()), (gv * w00).ravel())
        np.add.at(canvas, (y0.ravel(), x1.ravel()), (gv * w10).ravel())
        np.add.at(canvas, (y1.ravel(), x0.ravel()), (gv * w01).ravel())
        np.add.at(canvas, (y1.ravel(), x1.ravel()), (gv * w11).ravel())



    # 将来の前処理/後処理を挟み込むための薄いオーバーライド
    def __getitem__(self, idx: int):
        self._before_getitem(idx)
        out = super().__getitem__(idx)
        print('[debug]: FineTuningDataset_v1_2 __getitem__ idx=', idx)
        return self._after_getitem(idx, out)

    # フック: 必要になったらロジックを追加
    def _before_getitem(self, idx: int) -> None:
        # 例: インデックスに基づく動的パラメタ変更など
        return None

    def _after_getitem(self, idx: int, out: Tuple[Any, Any]) -> Tuple[Any, Any]:
        # 例: (img, gt) に対する追加入力生成・変換など
        return out

    # 例: 将来 GT 生成を差し替える場合の足場（現状は親をそのまま利用）
    # def generate_gt_heatmaps(self, *args: Any, **kwargs: Any):
    #     return super().generate_gt_heatmaps(*args, **kwargs)