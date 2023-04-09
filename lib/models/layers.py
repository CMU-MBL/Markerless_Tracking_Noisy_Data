import torch
from torch import nn

from einops import rearrange
from timm.models.layers import DropPath

"""
Pre-defined layers
"""


class MLP(nn.Module):
    """
    Reference from PoseFormer, ICCV 2021
    https://github.com/zczcwh/PoseFormer
    """

    def __init__(self, in_features, hidden_features=None, out_features=None,
                 act_layer=nn.GELU, drop=0.):
        super(MLP, self).__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features, bias=False)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features, bias=False)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class MultiHeadSelfAttention(nn.Module):
    """
    Reference from PoseFormer, ICCV 2021
    https://github.com/zczcwh/PoseFormer
    """

    def __init__(self, dim, num_heads, qkv_bias, qk_scale, attn_drop, proj_drop):
        super(MultiHeadSelfAttention, self).__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** (-0.5)

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim, bias=qkv_bias)
        self.proj_drop = nn.Dropout(proj_drop)


    def forward(self, x, mask=None):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C//self.num_heads
                                  ).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        attn = (q @ k.transpose(-2, -1)) * self.scale
        if mask is not None:
            attn.masked_fill_(mask.unsqueeze(1), -1e9)
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)

        return x, attn


class MultiHeadSelfAttentionBlock(nn.Module):
    """
    Reference from PoseFormer, ICCV 2021
    https://github.com/zczcwh/PoseFormer
    """

    def __init__(self, dim, num_heads, mlp_ratio=2., qkv_bias=True,
                 qk_scale=None, drop=0., attn_drop=0., drop_path=0.,
                 act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        super(MultiHeadSelfAttentionBlock, self).__init__()
        self.norm1 = norm_layer(dim)
        self.attn = MultiHeadSelfAttention(
            dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale,
            attn_drop=attn_drop, proj_drop=drop)
        self.drop_path = DropPath(drop_path)
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = MLP(in_features=dim, hidden_features=mlp_hidden_dim,
                       act_layer=act_layer, drop=drop)

    def forward(self, x, mask=None, y=None):
        identity = x
        out, attn = self.attn(self.norm1(x), mask)
        out = self.norm1(out)
        out += identity
        out = out + self.drop_path(self.mlp(self.norm2(out)))

        return out, attn


class MultiHeadAttention(nn.Module):

    def __init__(self, dim, num_heads=8, qkv_bias=True, qk_scale=None,
                 attn_drop=0., proj_drop=0.):
        super(MultiHeadAttention, self).__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** (-0.5)

        self.fcq = nn.Linear(dim, dim, bias=qkv_bias)
        self.fck = nn.Linear(dim, dim, bias=qkv_bias)
        self.fcv = nn.Linear(dim, dim, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim, bias=False)
        self.proj_drop = nn.Dropout(proj_drop)


    def forward(self, x, y, mask=None):
        B, Nx, C = x.shape
        _, Ny, C = y.shape
        q = self.fcq(x).reshape(B, Nx, self.num_heads, C // self.num_heads
                                ).permute(0, 2, 1, 3)
        k = self.fck(y).reshape(B, Ny, self.num_heads, C // self.num_heads
                                ).permute(0, 2, 1, 3)
        v = self.fcv(y).reshape(B, Ny, self.num_heads, C // self.num_heads
                                ).permute(0, 2, 1, 3)
        attn = (q @ k.transpose(-2, -1)) * self.scale
        if mask is not None:
            attn.masked_fill_(mask.unsqueeze(1), -1e9)
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, Nx, C)
        x = self.proj(x)
        x = self.proj_drop(x)

        return x, attn


class MultiHeadAttentionBlock(nn.Module):

    def __init__(self, dim, num_heads, mlp_ratio=2., qkv_bias=True,
                 qk_scale=None, drop=0., attn_drop=0., drop_path=0.,
                 act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        super(MultiHeadAttentionBlock, self).__init__()
        self.norm1x = norm_layer(dim)
        self.norm1y = norm_layer(dim)
        self.attn = MultiHeadAttention(
            dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale,
            attn_drop=attn_drop, proj_drop=drop)
        self.drop_path = DropPath(drop_path)
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = MLP(in_features=dim, hidden_features=mlp_hidden_dim,
                       act_layer=act_layer, drop=drop)

    def forward(self, x, y, mask=None):
        identity = x
        out, attn = self.attn(self.norm1x(x), self.norm1y(y), mask)
        out = identity + self.drop_path(out)
        out += identity
        out = out + self.drop_path(self.mlp(self.norm2(out)))
        return out, attn