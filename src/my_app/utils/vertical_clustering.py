import numpy as np
from skimage.segmentation import watershed
from skimage.feature import peak_local_max
from skimage.morphology import label
from scipy import ndimage as ndi
from matplotlib import pyplot as plt
import numpy as np, cv2

class VerticalClustering:
    def __init__(self):
        pass

    def process(self, org_image):
        labels = self.cluster_vertical_columns_adaptive(org_image)
        return labels

    def get_watershed_image(self, target_image):
        '''
        ウォータシェッドアルゴリズムによるセグメンテーションしたラベルマップを返す。
        - input -
        target_image: (H, W) の2次元画像テンソル
        - output -
        labels_ws: (H, W) のラベルマップ
        ピクセルの要素がラベルのIDを表している。0は背景。
        '''

        # arr: (H, W) の2値画像
        arr = target_image.cpu().numpy()
        distance = ndi.distance_transform_edt(arr > 0)

        # 局所最大値の座標リストを取得
        coordinates = peak_local_max(distance, labels=(arr > 0), footprint=np.ones((3, 3)))

        # マーカー画像を作成
        markers = np.zeros_like(distance, dtype=np.int32)
        for i, (y, x) in enumerate(coordinates, 1):
            markers[y, x] = i

        # ウォーターシェッド
        labels_ws = watershed(-distance, markers, mask=(arr > 0))
        return labels_ws

    def show_watershed_image(self, target_image):
        '''
        ウォータシェッドアルゴリズムによるセグメンテーションしたラベルマップを返す。元画像と出力を並べて表示する。
        - input -
        target_image: (H, W) の2次元画像テンソル
        - output -
        labels_ws: (H, W) のラベルマップ
        ピクセルの要素がラベルのIDを表している。0は背景。
        '''
        labels_ws = self.get_watershed_image(target_image)
        plt.subplot(1,2,1)
        plt.title("Original Image")
        plt.imshow(target_image.cpu().numpy())
        plt.subplot(1,2,2)
        plt.imshow(labels_ws, cmap='nipy_spectral')
        plt.title("Watershed result")
        plt.show()
        return labels_ws

    def show_watershed_combined_image(self, target_image):
        '''
        ウォータシェッドアルゴリズムによるセグメンテーションしたラベルマップを返す。元画像と出力を重ねて表示する。
        - input -
        target_image: (H, W) の2次元画像テンソル
        - output -
        labels_ws: (H, W) のラベルマップ
        ピクセルの要素がラベルのIDを表している。0は背景。
        '''
        labels_ws = self.get_watershed_image(target_image)
        combined_image = (target_image.cpu().numpy() / 2) + (labels_ws / 1000)
        plt.imshow(combined_image, cmap='nipy_spectral')
        plt.title("Combined Image")
        plt.show()
        return labels_ws

    def get_combine_image(self, image_A, image_B):
        combined_image = (image_A / 2) + (image_B / 2)
        return combined_image

    def show_combine_image(self, image_A, image_B):
        combined_image = self.get_combine_image(image_A, image_B)
        plt.imshow(combined_image, cmap='nipy_spectral')
        plt.title("Combined Image")
        plt.show()

    def fill_vertical_small_gaps(self, bin_img: np.ndarray, gap_limit: int):
        """
        列幅を太らせず、縦方向の 0 の短い区間だけ 1 で埋める。
        gap_limit 以下の 0 run を前後が1なら埋める。
        """
        h, w = bin_img.shape
        out = bin_img.copy()
        for x in range(w):
            col = out[:, x]
            y = 0
            while y < h:
                # 1 の run をスキップ
                while y < h and col[y] != 0:
                    y += 1
                start = y
                # 0 の run
                while y < h and col[y] == 0:
                    y += 1
                end = y
                # 前後が1 かつ 長さ <= gap_limit なら埋める
                if end > start and (end - start) <= gap_limit:
                    if start > 0 and end < h and col[start-1] != 0 and col[end] != 0:
                        col[start:end] = 255
            out[:, x] = col
        return out

    def cluster_vertical_columns_adaptive(self, 
        heatmap: np.ndarray,
        thresh: float = None,
        min_area: int = 30,
        connectivity: int = 4,
        # モルフォロジ制御
        mode: str = "closing",           # "dilation" | "closing" | "none"
        vertical_len: int = 25,
        iterations: int = 1,
        multi_vertical_lens: list = None, # 例 [10,25,40] で段階的
        # ランレングスギャップ埋め
        gap_limit: int = 12,            # 例 12: その長さ以下の縦穴だけ埋める
        use_otsu: bool = True
    ):
        """
        ギャップ埋め強度を段階・方式で調整。
        優先順位: gap_limit による細穴埋め → (multi or 単一) モルフォロジ → ラベリング
        """
        img = heatmap.astype(np.float32)
        if img.max() > 0:
            norm = img / img.max()
        else:
            norm = img
        u8 = (norm * 255).clip(0,255).astype(np.uint8)

        # 1) 二値化
        if thresh is None:
            if use_otsu:
                _, bin_img = cv2.threshold(u8, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            else:
                t = np.quantile(u8, 0.3)
                _, bin_img = cv2.threshold(u8, t, 255, cv2.THRESH_BINARY)
        else:
            val = int(thresh*255) if thresh <= 1 else int(thresh)
            _, bin_img = cv2.threshold(u8, val, 255, cv2.THRESH_BINARY)

        # 2) ランレングスで小さい縦穴だけ埋める (列太り無し)
        if gap_limit is not None:
            bin_img = self.fill_vertical_small_gaps(bin_img, gap_limit)

        # 3) モルフォロジ (列全体をつなぐ強度調整)
        work = bin_img.copy()
        if multi_vertical_lens:
            for k in multi_vertical_lens:
                kernel = np.ones((k,1), np.uint8)
                if mode == "dilation":
                    work = cv2.dilate(work, kernel, iterations=1)
                elif mode == "closing":
                    work = cv2.morphologyEx(work, cv2.MORPH_CLOSE, kernel, iterations=1)
                # mode=="none" なら何もしない
        else:
            if mode != "none":
                kernel = np.ones((vertical_len,1), np.uint8)
                if mode == "dilation":
                    work = cv2.dilate(work, kernel, iterations=iterations)
                elif mode == "closing":
                    work = cv2.morphologyEx(work, cv2.MORPH_CLOSE, kernel, iterations=iterations)

        # 4) 連結成分
        n_labels, labels = cv2.connectedComponents(work, connectivity)

        # 5) 元二値(または小穴埋め後 bin_img)でマスク (細線形状へ戻したいならココを bin_img に)
        labels_refined = np.zeros_like(labels, np.int32)
        fg = bin_img > 0
        labels_refined[fg] = labels[fg]

        # 6) 小領域除去
        if min_area > 0:
            counts = np.bincount(labels_refined.ravel())
            remove_ids = np.where(counts < min_area)[0]
            for rid in remove_ids:
                if rid == 0: continue
                labels_refined[labels_refined == rid] = 0

        # 7) 再連番
        used = np.unique(labels_refined)
        used = used[used != 0]
        remap = {old:i+1 for i,old in enumerate(used)}
        for old,new in remap.items():
            labels_refined[labels_refined == old] = new
        return labels_refined