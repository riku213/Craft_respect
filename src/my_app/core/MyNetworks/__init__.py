from .MyUnet import UNet
from .DeepUnet import DeepUNet  

# __all__ を定義すると、'from my_app import *' でインポートされる対象を明示できる
__all__ = ['UNet', 'DeepUNet']