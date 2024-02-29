import torch
import torch.nn as nn
from einops.layers.torch import Rearrange
from swin_transformer import SwinTransformer

if __name__ == '__main__':
    x = torch.tensor([[[[1,2,3,4], [5,6,7,8],[9,10,11,12], [13,14,15,16]]]])
    #x_ = x.view([1,1,2,2,2,2])
    #x_ = x_.permute([0, 1, 2, 4, 3, 5]).reshape([1, 1, 4, 4])
    op = Rearrange('b c (h p1) (w p2) -> b c (h w) p1 p2', p1 = 2, p2 = 2)
    x_ = op(x)
    x_show = x_[0, 0, 1]
    print(x)
    print(x_show)