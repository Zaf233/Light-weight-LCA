import torch
from torch import nn, einsum
import torch.nn.functional as F

from einops import rearrange, repeat
from einops.layers.torch import Rearrange

# helpers

def pair(t):
    return t if isinstance(t, tuple) else (t, t)

# classes

class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn
    def forward(self, x, **kwargs):
        return self.fn(self.norm(x), **kwargs)

class FeedForward(nn.Module):
    'simple fully connected layer structure'
    def __init__(self, dim, hidden_dim, dropout = 0.):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout)
        )
    def forward(self, x):
        return self.net(x)

class Attention(nn.Module):
    "muti-head self attention"
    def __init__(self, dim, heads = 8, dim_head = 64, dropout = 0.):
        super().__init__()
        inner_dim = dim_head *  heads
        project_out = not (heads == 1 and dim_head == dim)

        self.heads = heads
        self.scale = dim_head ** -0.5

        self.attend = nn.Softmax(dim = -1)
        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias = False)

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        ) if project_out else nn.Identity()

    def forward(self, x):
        b, n, _, h = *x.shape, self.heads
        qkv = self.to_qkv(x).chunk(3, dim = -1)
        #扩充维度后在最后一个维度划分为3份。
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h = h), qkv)

        dots = einsum('b h i d, b h j d -> b h i j', q, k) * self.scale # softmax(Q乘以K的转置除以根号下dim_head)

        attn = self.attend(dots) # softmax

        out = einsum('b h i j, b h j d -> b h i d', attn, v)#这里的einsum就是矩阵乘法
        out = rearrange(out, 'b h n d -> b n (h d)')
        return self.to_out(out) # 由回到输入之前的维度

class Transformer(nn.Module):
    def __init__(self, dim, depth, heads, dim_head, mlp_dim, dropout = 0.):
        super().__init__()
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                PreNorm(dim, Attention(dim, heads = heads, dim_head = dim_head, dropout = dropout)),
                PreNorm(dim, FeedForward(dim, mlp_dim, dropout = dropout))
            ]))
    def forward(self, x):
        for attn, ff in self.layers:
            x = attn(x) + x
            x = ff(x) + x
        return x

class crossAttention(nn.Module):
    "muti-head self attention"
    def __init__(self, dim, heads = 8, dim_head = 64, dropout = 0.):
        super().__init__()
        inner_dim = dim_head *  heads
        project_out = not (heads == 1 and dim_head == dim)

        self.heads = heads
        self.scale = dim_head ** -0.5

        self.attend = nn.Softmax(dim = -1)
        self.to_qkv1 = nn.Linear(dim, inner_dim * 3, bias = False)
        self.to_qkv2 = nn.Linear(dim, inner_dim * 3, bias = False)

        self.to_out1 = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        ) if project_out else nn.Identity()

        self.to_out2 = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        ) if project_out else nn.Identity()

    def forward(self, x1, x2):
        b, n, _, h = *x1.shape, self.heads
        qkv1 = self.to_qkv1(x1).chunk(3, dim = -1)
        qkv2 = self.to_qkv2(x2).chunk(3, dim = -1)
        #扩充维度后在最后一个维度划分为3份。
        q1, k1, v1 = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h = h), qkv1)
        q2, k2, v2 = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h = h), qkv2)

        dots1 = einsum('b h i d, b h j d -> b h i j', q1, k2) * self.scale # softmax(Q乘以K的转置除以根号下dim_head)
        dots2 = einsum('b h i d, b h j d -> b h i j', q2, k1) * self.scale # softmax(Q乘以K的转置除以根号下dim_head)

        attn1 = self.attend(dots1) # softmax
        attn2 = self.attend(dots2) # softmax

        out1 = einsum('b h i j, b h j d -> b h i d', attn1, v2)#这里的einsum就是矩阵乘法
        out1 = rearrange(out1, 'b h n d -> b n (h d)')

        out2 = einsum('b h i j, b h j d -> b h i d', attn2, v1)#这里的einsum就是矩阵乘法
        out2 = rearrange(out2, 'b h n d -> b n (h d)')

        return self.to_out1(out1), self.to_out2(out2) # 由回到输入之前的维度

class crossTransformer(nn.Module):
    def __init__(self, dim, depth, heads, dim_head, mlp_dim, dropout = 0.):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                crossAttention(dim, heads = heads, dim_head = dim_head, dropout = dropout),
                PreNorm(dim, FeedForward(dim, mlp_dim, dropout = dropout))
            ]))
    def forward(self, x1, x2):
        for attn, ff in self.layers:
            x1_, x2_ = attn(self.norm(x1), self.norm(x2))
            x1_ = x1_ + x1
            x2_ = x2_ + x2
            x1 = ff(x1_) + x1_
            x2 = ff(x2_) + x2_
        return x1, x2

class ViT(nn.Module):
    def __init__(self, *, image_size, patch_size, dim, depth, heads, mlp_dim,
                 pool='cls', channels=3, dim_head=64, dropout=0., emb_dropout=0.):
        super().__init__()
        image_height, image_width = pair(image_size)
        patch_height, patch_width = pair(patch_size)
        # dim是图像网格化后每一个网格序列化后需要统一到一个相同的序列化长度
        # depth是transform里面encoder的堆叠数量
        # heads是multi-head attention里面的head数量， dim_head是每个head里面的qkv长度
        # mlp_dim是前向网络里面的隐藏层的tersor长度

        assert image_height % patch_height == 0 and image_width % patch_width == 0, 'Image dimensions must be divisible by the patch size.'

        num_patches = (image_height // patch_height) * (image_width // patch_width)
        # num_patches是将图片网格化后网格的数量
        patch_dim = channels * patch_height * patch_width
        # patch_dim是将图片网格化后每一个网格序列化的长度
        assert pool in {'cls', 'mean'}, 'pool type must be either cls (cls token) or mean (mean pooling)'

        self.to_patch_embedding = nn.Sequential(
            Rearrange('b c (h p1) (w p2) -> b (h w) (p1 p2 c)', p1 = patch_height, p2 = patch_width),
            nn.Linear(patch_dim, dim),
        ) # 输出的维度是（b, h*w, dim）

        self.pos_embedding = nn.Parameter(torch.randn(1, num_patches + 1, dim))
        self.cls_token = nn.Parameter(torch.randn(1, 1, dim))
        self.dropout = nn.Dropout(emb_dropout)

        self.transformer = Transformer(dim, depth, heads, dim_head, mlp_dim, dropout)

        self.pool = pool
        self.to_latent = nn.Identity() # nn.Identity是用来占位的

    def forward(self, img):
        x = self.to_patch_embedding(img)
        b, n, _ = x.shape
        # n是每一个网格的h*w
        cls_tokens = repeat(self.cls_token, '() n d -> b n d', b = b)
        # cls_tokens在batch维度进行扩充
        x = torch.cat((cls_tokens, x), dim=1)
        x += self.pos_embedding[:, :(n + 1)]
        x = self.dropout(x)

        x = self.transformer(x)

        x = x.mean(dim=1) if self.pool == 'mean' else x[:, 0]

        x = self.to_latent(x)
        return x

class mySelfTransformer(nn.Module):
    def __init__(self, *, image_size, patch_size, dim, depth, heads, mlp_dim, channels=3, dim_head=64, dropout=0., emb_dropout=0.):
        super().__init__()
        image_height, image_width = pair(image_size)
        patch_height, patch_width = pair(patch_size)
        # dim是图像网格化后每一个网格序列化后需要统一到一个相同的序列化长度
        # depth是transform里面encoder的堆叠数量
        # heads是multi-head attention里面的head数量， dim_head是每个head里面的qkv长度
        # mlp_dim是前向网络里面的隐藏层的tersor长度

        assert image_height % patch_height == 0 and image_width % patch_width == 0, 'Image dimensions must be divisible by the patch size.'

        num_patches = (image_height // patch_height) * (image_width // patch_width)
        # num_patches是将图片网格化后网格的数量
        patch_dim = channels * patch_height * patch_width
        # patch_dim是将图片网格化后每一个网格序列化的长度

        self.to_patch_embedding = nn.Sequential(
            Rearrange('b c (h p1) (w p2) -> b (h w) (p1 p2 c)', p1 = patch_height, p2 = patch_width),
            nn.Linear(patch_dim, dim),
        ) # 输出的维度是（b, h*w, dim）

        self.pos_embedding = nn.Parameter(torch.randn(1, num_patches, dim))
        self.dropout = nn.Dropout(emb_dropout)
        self.transformer = Transformer(dim, depth, heads, dim_head, mlp_dim, dropout)

    def forward(self, img):
        x = self.to_patch_embedding(img)
        b, n, _ = x.shape
        # n是每一个网格的h*w
        x += self.pos_embedding[:, :n]
        x = self.dropout(x)

        x = self.transformer(x)
        return x

class mySelfTransformer_seq(nn.Module):
    def __init__(self, *, num_patches, dim, depth, heads, mlp_dim, dim_head=64, dropout=0., emb_dropout=0.):
        super().__init__()
        self.pos_embedding = nn.Parameter(torch.randn(1, num_patches, dim))
        self.dropout = nn.Dropout(emb_dropout)
        self.transformer = Transformer(dim, depth, heads, dim_head, mlp_dim, dropout)

    def forward(self, seq):
        b, n, _ = seq.shape
        # n是每一个网格的h*w
        seq += self.pos_embedding[:, :n]
        seq = self.dropout(seq)

        seq = self.transformer(seq)
        return seq

class myCrossTransformer(nn.Module):
    def __init__(self, *, image_size, patch_size, dim, depth, heads, mlp_dim, channels=3, dim_head=64, dropout=0.,
                 emb_dropout=0.):
        super().__init__()
        image_height, image_width = pair(image_size)
        patch_height, patch_width = pair(patch_size)
        # dim是图像网格化后每一个网格序列化后需要统一到一个相同的序列化长度
        # depth是transform里面encoder的堆叠数量
        # heads是multi-head attention里面的head数量， dim_head是每个head里面的qkv长度
        # mlp_dim是前向网络里面的隐藏层的tersor长度

        assert image_height % patch_height == 0 and image_width % patch_width == 0, 'Image dimensions must be divisible by the patch size.'

        num_patches = (image_height // patch_height) * (image_width // patch_width)
        # num_patches是将图片网格化后网格的数量
        patch_dim = channels * patch_height * patch_width
        # patch_dim是将图片网格化后每一个网格序列化的长度
        self.to_patch_embedding = nn.Sequential(
            Rearrange('b c (h p1) (w p2) -> b (h w) (p1 p2 c)', p1 = patch_height, p2 = patch_width),
            nn.Linear(patch_dim, dim),
        )  # 输出的维度是（b, h*w, dim）
        self.pos_embedding1 = nn.Parameter(torch.randn(1, num_patches, dim))
        self.pos_embedding2 = nn.Parameter(torch.randn(1, num_patches, dim))
        self.dropout = nn.Dropout(emb_dropout)
        self.transformer = crossTransformer(dim, depth, heads, dim_head, mlp_dim, dropout)

    def forward(self, img1, img2):
        x1 = self.to_patch_embedding(img1)
        x2 = self.to_patch_embedding(img2)
        b, n, _ = x1.shape
        # n是每一个网格的h*w
        x1 = x1 + self.pos_embedding1[:, :n]
        x2 = x2 + self.pos_embedding2[:, :n]
        x1, x2 = self.transformer(x1, x2)
        x = torch.cat([x1, x2], dim=1)
        return x


if __name__ == "__main__":
    test_input1 = torch.randn((2, 128, 8, 16))
    test_input2 = torch.randn((2, 128, 8, 16))
    model = mySelfTransformer(image_size=(8, 16), channels=128, patch_size=(1, 1), dim=128, depth=6, heads=8, mlp_dim=64)
    trans, rot = model(test_input1)
    print(trans.shape)
    print(rot.shape)
