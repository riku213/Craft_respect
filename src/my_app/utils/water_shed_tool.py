import numpy as np
from skimage.segmentation import watershed
from skimage.measure import label
from skimage.color import label2rgb
from matplotlib import pyplot as plt

class WaterShedTool:
    def water_shed_segmentation(self, pred):
        focused_pred = pred.numpy()
        prob = focused_pred.astype(np.float32)
        # マーカー作成（閾値は用途に合わせて調整）
        fg_thresh = 0.5
        mask_thresh = 0.1
        markers = label(prob > fg_thresh)
        labels = watershed(-prob, markers=markers, mask=(prob > mask_thresh))
        return labels
    
    def show_label(self, label):
        plt.title('label')
        plt.imshow(label, cmap=('tab20'))
        plt.show()

    def show_overlay(self, img, label):
        nump_img = img.permute(1,2,0).numpy()
        overlay = label2rgb(label, image=nump_img, bg_label=0, alpha=0.4)
        plt.title('overlay')
        plt.imshow(overlay)
        plt.show()