""" Vision Transformer (ViT) in PyTorch

A PyTorch implement of Vision Transformers as described in:

'An Image Is Worth 16 x 16 Words: Transformers for Image Recognition at Scale'
    - https://arxiv.org/abs/2010.11929

`How to train your ViT? Data, Augmentation, and Regularization in Vision Transformers`
    - https://arxiv.org/abs/2106.10270

The official jax code is released and available at https://github.com/google-research/vision_transformer

Acknowledgments:
* The paper authors for releasing code and weights, thanks!
* I fixed my class token impl based on Phil Wang's https://github.com/lucidrains/vit-pytorch ... check it out
for some einops/einsum fun
* Simple transformer style inspired by Andrej Karpathy's https://github.com/karpathy/minGPT
* Bert reference code checks against Huggingface Transformers and Tensorflow Bert

Hacked together by / Copyright 2020, Ross Wightman
"""
import math
import logging
from functools import partial
from collections import OrderedDict

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint

from timm.data import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD, IMAGENET_INCEPTION_MEAN, IMAGENET_INCEPTION_STD
from .helpers import build_model_with_cfg, resolve_pretrained_cfg, named_apply, adapt_input_conv, checkpoint_seq
from .layers import PatchEmbed, Mlp, DropPath, trunc_normal_, lecun_normal_, PatchEmbedDCT, PatchEmbedDCT_P
from .registry import register_model

import torchjpeg.dct as dct
import torch.fft
import pywt
import numpy as np

_logger = logging.getLogger(__name__)


def _cfg(url='', **kwargs):
    return {
        'url': url,
        'num_classes': 1000, 'input_size': (3, 224, 224), 'pool_size': None,
        'crop_pct': .9, 'interpolation': 'bicubic', 'fixed_input_size': True,
        'mean': IMAGENET_INCEPTION_MEAN, 'std': IMAGENET_INCEPTION_STD,
        'first_conv': 'patch_embed.proj', 'classifier': 'head',
        **kwargs
    }


class Attention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, attn_drop=0., proj_drop=0.):
        super().__init__()
        assert dim % num_heads == 0, 'dim should be divisible by num_heads'
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)   # make torchscript happy (cannot use tensor as tuple)

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class DctMLP(nn.Module):
    def __init__(self, dim=384, mlp_ratio=4, act_layer=nn.GELU, drop=0, reduce_ratio_C=100, DCT_MAT=False):
        super().__init__()

        self.reduce_C = dim-math.ceil(dim*reduce_ratio_C/100.)
        
        Dim = dim-self.reduce_C
        mlp_hidden_Dim = int(Dim*mlp_ratio)
        self.mlp = Mlp(in_features=Dim, hidden_features=mlp_hidden_Dim, act_layer=act_layer, drop=drop)

    def forward(self, x):
        B, N, C = x.shape        
        x = x.transpose(1,2) # B, C, N
        dct_mat = dct_matrix(C,C-self.reduce_C).to(x.device)
        x = torch.matmul(dct_mat, x)
        x = x.transpose(1,2) # B, N, C
        x = self.mlp(x)            
        x = x.transpose(1,2) # B, C, N
        x = torch.matmul(dct_mat.t(), x)
        x = x.transpose(1,2) # B, N, C      
        
        return x

class GlobalFilter(nn.Module):
    def __init__(self, dim, h=14, w=8, reduce_n=0):
        super().__init__()
        self.complex_weight = nn.Parameter(torch.randn(h, w//2+1, dim, 2, dtype=torch.float32) * 0.02)
        self.w = w
        self.h = h
        self.reduce_n = reduce_n

    def forward(self, x, spatial_size=None):
        B, N, C = x.shape
        if spatial_size is None:
            a = b = int(math.sqrt(N))
        else:
            a, b = spatial_size
        x = x.view(B, a, b, C)
        x = x.to(torch.float32)

        if self.reduce_n > 0:
            # gf_spatial = torch.fft.irfft2(x, dim=(1, 2), s=(a, b))
            gf_complex = torch.fft.fft2(x, dim=(1, 2))
            gf_complex = torch.fft.fftshift(gf_complex, dim=(1, 2))
            gf_complex = gf_complex[:, self.reduce_n//2:-self.reduce_n//2, self.reduce_n//2:-self.reduce_n//2, :]
            gf_complex = torch.fft.fftshift(gf_complex, dim=(1, 2))
            gf_spatial = torch.fft.irfft2(gf_complex, dim=(1, 2), s=(a-self.reduce_n,b-self.reduce_n))
            x = torch.fft.rfft2(gf_spatial, dim=(1, 2), norm='ortho')
        else:
            x = torch.fft.rfft2(x, dim=(1, 2), norm='ortho')
                
        weight = torch.view_as_complex(self.complex_weight)
        x = x * weight
        x = torch.fft.irfft2(x, s=(a-self.reduce_n, b-self.reduce_n), dim=(1, 2), norm='ortho')
        
        if self.reduce_n > 0:
            x = x.reshape(B, (a-self.reduce_n)**2, C)
        else:
            x = x.reshape(B, N, C)

        return x
    
class GlobalFilter_CF_dft(nn.Module):
    def __init__(self, dim, h=14, w=14, reduce_c=0):
        super().__init__()
        self.w = w
        self.h = h
        self.reduce_c = reduce_c
        self.weight_c = nn.Parameter(torch.ones(h*w, dim//2+1, 2, dtype=torch.float32)*0.02)

    def forward(self, x, spatial_size=None):
        B, N, C = x.shape
        x = x.to(torch.float32)
        # x = dct1(x)
        x = torch.fft.rfft(x, dim=2, norm='ortho')
        if self.reduce_c > 0:
            # print('reduce_c: ', self.reduce_c)
            x = x[:,:,:-self.reduce_c]
        weight_c = torch.view_as_complex(self.weight_c)
        x = x * weight_c
        # x = idct1(x)
        x = torch.fft.irfft(x, dim=2, norm='ortho')

        return x
    
class GlobalDCTFilter(nn.Module):
    def __init__(self, dim, h=14, w=14, reduce_n=0):
        super().__init__()
        self.w = w
        self.h = h
        self.reduce_n = reduce_n
        Qtable = qtable(h,w)
        self.weight = nn.Parameter(Qtable.expand(dim, h, w).clone())

    def forward(self, x, spatial_size=None):
        B, N, C = x.shape
        a = b = int(math.sqrt(N))
                                # B C b a       B C a b       B 1 C a b
        x = x.view(B, a, b, C).transpose(1,3).transpose(2,3).unsqueeze(1)

        x = x.to(torch.float32)

        x = dct.block_dct(x)
        if self.reduce_n > 0:
            x = x[:,:,:,:-self.reduce_n,:-self.reduce_n]
        
        x = x * self.weight #.clone()
        
        x = dct.block_idct(x)
            # B C a b      B b a C        B a b C
        x = x.squeeze(1).transpose(1,3).transpose(1,2)

        if self.reduce_n > 0:
            x = x.reshape(B, (a-self.reduce_n)**2, C)
        else:
            x = x.reshape(B, N, C)

        return x

class GlobalFilter_CF(nn.Module):
    def __init__(self, dim, h=14, w=14, reduce_c=0):
        super().__init__()
        self.w = w
        self.h = h
        self.reduce_c = reduce_c
        self.weight_c = nn.Parameter(torch.ones(h*w, dim, dtype=torch.float32))

    def forward(self, x, spatial_size=None):
        B, N, C = x.shape
        x = x.to(torch.float32)
        # x = dct1(x)
        x = dct_2_1d(x)
        if self.reduce_c > 0:
            # print('reduce_c: ', self.reduce_c)
            x = x[:,:,:-self.reduce_c]
        x = x * self.weight_c
        # x = idct1(x)
        x = idct_2_1d(x)

        return x
    
class GlobalDCTFilter_1d(nn.Module):
    def __init__(self, dim, h=14, w=14, reduce_n=0, reduce_ratio=-1):
        super().__init__()
        self.w = w
        self.h = h
        self.reduce_n = reduce_n
        self.reduce_ratio = reduce_ratio
        Qtable = qtable(h,w)
        self.weight = nn.Parameter(torch.randn(dim, h*w, dtype=torch.float32)*0.02)

    def forward(self, x):
        B, N, C = x.shape
        x = x.transpose(1,2)
        x = x.to(torch.float32)
        x = dct_2_1d(x)
        if self.reduce_ratio >0:
            reduce_N = N-math.ceil(N*self.reduce_ratio/100.)      #[50] 98(98) -> 49(49) -> 24(25)  //   [60]   //   [70]  58(138) -> 41(97) -> 29(68)
            if reduce_N >0:
                # x = x[:,:,:-reduce_N]    
                # x = x[:,:,reduce_N:]
                x = x[:,:,::2]
        elif self.reduce_n > 0:
            # 196 = 14*14
            if self.reduce_n == 1:                 # 27(169) -> 25(144) -> 23(121)
                if N==196: reduce_N = 27
                elif N==169: reduce_N = 25
                elif N==144: reduce_N = 23
                else: ('error in reduce_N')
            elif self.reduce_n == 2:                 # -26.5% 52(144) -> -30.5% 44(100) -> -36% 36(64)
                if N==196: reduce_N = 52
                elif N==144: reduce_N = 44
                elif N==100: reduce_N = 36
                else: ('error in reduce_N')
            elif self.reduce_n == 3:                 # 75(121) -> 57(64) -> 39(25)
                if N==196: reduce_N = 75
                elif N==121: reduce_N = 57
                elif N==64: reduce_N = 39
                else: ('error in reduce_N')
            elif self.reduce_n == 4:                 # 96(100) -> 64(36) -> 32(4)
                if N==196: reduce_N = 96
                elif N==100: reduce_N = 64
                elif N==64: reduce_N = 32
                else: ('error in reduce_N')
            x = x[:,:,:-reduce_N]    
        
        x = x * self.weight
        x = idct_2_1d(x)
        x = x.transpose(1,2)

        return x
    
class LayerScale(nn.Module):
    def __init__(self, dim, init_values=1e-5, inplace=False):
        super().__init__()
        self.inplace = inplace
        self.gamma = nn.Parameter(init_values * torch.ones(dim))

    def forward(self, x):
        return x.mul_(self.gamma) if self.inplace else x * self.gamma


class Block(nn.Module):
    def __init__(
            self, dim, num_heads, mlp_ratio=4., qkv_bias=False, drop=0., attn_drop=0., init_values=None, h=14, w=14, GCAM=False, dftGF=False,
            drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm, dctMSA=False, dctGF=False, dctGF_1d=False, reduce_n=0, reduce_ratio=-1, skip_connect='vit', GCM=False):
        super().__init__()
        self.norm1 = norm_layer(dim)
        if dctGF == True:
            print('dctGF')
            self.attn = GlobalDCTFilter(dim, h=h, w=w, reduce_n=reduce_n)
            if GCM == True:
                self.attn2 = GlobalFilter_CF(dim, h=h, w=w)
                self.norm3 = norm_layer(dim)
                self.drop_path3 = DropPath(drop_path) if drop_path > 0. else nn.Identity()
                self.ls3 = LayerScale(dim, init_values=init_values) if init_values else nn.Identity()
        elif dftGF == True:
            print('dftGF')
            self.attn = GlobalFilter(dim, h=h, w=w, reduce_n=reduce_n)
            if GCM == True:
                self.attn2 = GlobalFilter_CF_dft(dim, h=h, w=w)
                self.norm3 = norm_layer(dim)
                self.drop_path3 = DropPath(drop_path) if drop_path > 0. else nn.Identity()
                self.ls3 = LayerScale(dim, init_values=init_values) if init_values else nn.Identity()
        elif dctGF_1d == True:
            print('dctGF_1d')
            self.attn = GlobalDCTFilter_1d(dim, h=h, w=w, reduce_n=reduce_n, reduce_ratio=reduce_ratio)
            if GCM == True:
                self.attn2 = GlobalFilter_CF(dim, h=h, w=w)
                self.norm3 = norm_layer(dim)
                self.drop_path3 = DropPath(drop_path) if drop_path > 0. else nn.Identity()
                self.ls3 = LayerScale(dim, init_values=init_values) if init_values else nn.Identity()
        else:
            if GCAM == True:
                print('GCAM(GF-CGF-Att-Mlp) block')
                self.attn = Attention(dim, num_heads=num_heads, qkv_bias=qkv_bias, attn_drop=attn_drop, proj_drop=drop)
                self.GF = GlobalDCTFilter_1d(dim, h=h, w=w)
                self.CGF = GlobalFilter_CF(dim, h=h, w=w)
                self.ls3 = LayerScale(dim, init_values=init_values) if init_values else nn.Identity()
                self.ls4 = LayerScale(dim, init_values=init_values) if init_values else nn.Identity()
                self.drop_path3 = DropPath(drop_path) if drop_path > 0. else nn.Identity()
                self.drop_path4 = DropPath(drop_path) if drop_path > 0. else nn.Identity()
                self.norm3 = norm_layer(dim)
                self.norm4 = norm_layer(dim)
            else:
                print('Basic Attention')
                self.attn = Attention(dim, num_heads=num_heads, qkv_bias=qkv_bias, attn_drop=attn_drop, proj_drop=drop)
        self.ls1 = LayerScale(dim, init_values=init_values) if init_values else nn.Identity()
        # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
        self.drop_path1 = DropPath(drop_path) if drop_path > 0. else nn.Identity()

        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        if dctMLP == True and reduce_ratio_C>0:
            self.mlp = DctMLP(dim=dim, mlp_ratio=mlp_ratio, act_layer=act_layer, drop=drop, reduce_ratio_C=reduce_ratio_C)
        else:
            self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)
        self.ls2 = LayerScale(dim, init_values=init_values) if init_values else nn.Identity()
        if skip_connect=='vit':
            self.drop_path2 = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.dctMSA=dctMSA
        self.reduce_n=reduce_n
        self.reduce_ratio=reduce_ratio
        self.skip_connect=skip_connect
        self.GCM=GCM
        self.GCAM=GCAM

    def forward(self, x):
        if self.skip_connect=='vit':
            if self.GCM==True: 
                if self.reduce_n >0:
                    B, N, C = x.shape
                    a = b = int(math.sqrt(N)) # B C b a       B C a b       B 1 C a b                                            
                    x_res = x.view(B, a, b, C).transpose(1,3).transpose(2,3).unsqueeze(1)
                    x_res = x_res.to(torch.float32)
                    x_res = dct.block_dct(x_res)
                    x_res = x_res[:,:,:,:-self.reduce_n,:-self.reduce_n]
                    x_res = dct.block_idct(x_res)
                    x_res = x_res.squeeze(1).transpose(1,3).transpose(1,2)
                    x_res = x_res.reshape(B, (a-self.reduce_n)**2, C)

                    x = x_res + self.drop_path1(self.ls1(self.attn(self.norm1(x))))
                elif self.reduce_ratio >0:
                    B, N, C = x.shape
                    reduce_N = N-math.ceil(N*self.reduce_ratio/100.)
                    x_res = x.transpose(1,2)
                    x_res = x_res.to(torch.float32)
                    x_res = dct_2_1d(x_res)
                    if reduce_N >0:
                        x_res = x_res[:,:,:-reduce_N]    
                    x_res = idct_2_1d(x_res)
                    x_res = x_res.transpose(1,2)

                    x = x_res + self.drop_path1(self.ls1(self.attn(self.norm1(x))))
                else:
                    x = x + self.drop_path1(self.ls1(self.attn(self.norm1(x))))
                x = x + self.drop_path3(self.ls3(self.attn2(self.norm3(x))))
                x = x + self.drop_path2(self.ls2(self.mlp(self.norm2(x))))
            else:
                if self.reduce_n > 0:
                    y = self.drop_path1(self.ls1(self.attn(self.norm1(x))))

                    B, N, C = x.shape
                    # dct
                    a = b = int(math.sqrt(N))
                                    # B C b a       B C a b       B 1 C a b
                    x = x.view(B, a, b, C).transpose(1,3).transpose(2,3).unsqueeze(1)
                    # x = x.to(torch.float32)
                    x = dct.block_dct(x)
                    x = x[:,:,:,:-self.reduce_n,:-self.reduce_n]
                    x = dct.block_idct(x)
                        # B C a b      B b a C        B a b C
                    x = x.squeeze(1).transpose(1,3).transpose(1,2).reshape(B, (a-self.reduce_n)**2, C)

                    x = x + y
                    x = x + self.drop_path2(self.ls2(self.mlp(self.norm2(x))))
                else:
                    if self.GCAM == True:
                        x = x + self.drop_path3(self.ls3(self.GF(self.norm3(x))))
                        x = x + self.drop_path4(self.ls4(self.CGF(self.norm4(x))))
                        x = x + self.drop_path1(self.ls1(self.attn(self.norm1(x))))
                        x = x + self.drop_path2(self.ls2(self.mlp(self.norm2(x))))
                    else:
                        x = x + self.drop_path1(self.ls1(self.attn(self.norm1(x))))
                        x = x + self.drop_path2(self.ls2(self.mlp(self.norm2(x))))
        elif self.skip_connect=='layer':
            if self.reduce_n > 0:
                y = self.ls1(self.attn(self.norm1(x)))

                B, N, C = x.shape
                # dct
                a = b = int(math.sqrt(N))
                                # B C b a       B C a b       B 1 C a b
                x = x.view(B, a, b, C).transpose(1,3).transpose(2,3).unsqueeze(1)
                # x = x.to(torch.float32)
                x = dct.block_dct(x)
                x = x[:,:,:,:-self.reduce_n,:-self.reduce_n]
                x = dct.block_idct(x)
                    # B C a b      B b a C        B a b C
                x = x.squeeze(1).transpose(1,3).transpose(1,2).reshape(B, (a-self.reduce_n)**2, C)

                x = x + self.drop_path1(self.ls2(self.mlp(self.norm2(y))))
            else:
                x = x + self.drop_path1(self.ls2(self.mlp(self.norm2(self.ls1(self.attn(self.norm1(x)))))))

        return x



class VisionTransformer(nn.Module):
    """ Vision Transformer

    A PyTorch impl of : `An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale`
        - https://arxiv.org/abs/2010.11929
    """

    def __init__(
            self, img_size=224, patch_size=16, in_chans=3, num_classes=1000, global_pool='token',
            embed_dim=768, depth=12, num_heads=12, mlp_ratio=4., qkv_bias=True, representation_size=None, reduce_ratio=-1, dftGF=False, reduce_ratio_C=-1, dctMLP=False,
            drop_rate=0., attn_drop_rate=0., drop_path_rate=0., weight_init='', init_values=None, dctGF_1d=False, GCM=False, dctGF_layer=-1, L1st=False, GCAM=False, embed_layer=PatchEmbed, norm_layer=None, act_layer=None, block_fn=Block, model_input_size=224, dctMSA=False, reduce_n=0, dctGF=False, skip_connect='vit'):
        """
        Args:
            img_size (int, tuple): input image size
            patch_size (int, tuple): patch size
            in_chans (int): number of input channels
            num_classes (int): number of classes for classification head
            global_pool (str): type of global pooling for final sequence (default: 'token')
            embed_dim (int): embedding dimension
            depth (int): depth of transformer
            num_heads (int): number of attention heads
            mlp_ratio (int): ratio of mlp hidden dim to embedding dim
            qkv_bias (bool): enable bias for qkv if True
            representation_size (Optional[int]): enable and set representation layer (pre-logits) to this value if set
            drop_rate (float): dropout rate
            attn_drop_rate (float): attention dropout rate
            drop_path_rate (float): stochastic depth rate
            weight_init: (str): weight init scheme
            init_values: (float): layer-scale init values
            embed_layer (nn.Module): patch embedding layer
            norm_layer: (nn.Module): normalization layer
            act_layer: (nn.Module): MLP activation layer
        """
        super().__init__()
        assert global_pool in ('', 'avg', 'token')
        norm_layer = norm_layer or partial(nn.LayerNorm, eps=1e-6)
        act_layer = act_layer or nn.GELU
        self.YUV = YUV
        self.num_classes = num_classes
        self.global_pool = global_pool
        self.num_features = self.embed_dim = embed_dim  # num_features for consistency with other models
        if global_pool == 'avg':
            self.num_tokens = 0
        else:
            self.num_tokens = 1
        self.grad_checkpointing = False
        
        self.patch_embed = embed_layer(
            img_size=img_size, patch_size=patch_size, in_chans=in_chans, embed_dim=embed_dim)
            
        num_patches = self.patch_embed.num_patches
        if global_pool == 'toekn':
            self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + self.num_tokens, embed_dim))
        self.pos_drop = nn.Dropout(p=drop_rate)

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]  # stochastic depth decay rule
        
        h = img_size // patch_size
        w = h
        
        if reduce_n>0 or dctGF==True or dctGF_1d==True or reduce_ratio>0 or dftGF==True:
            self.blocks = nn.ModuleList()
            # dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depth))]  # stochastic depth decay rule
            # cur = 0
            #----------------------------------------------------------------------------------------------------------------------------------
            if L1st==True and GCM==True:
                stage_num = 4
                layers_in_stage = 3
                for i in range(stage_num):
                    print('using dct down-sampling block')
                    if reduce_n>0:
                        reduced_h = h-reduce_n*i
                        reduced_w = w-reduce_n*i
                    if reduce_ratio>0:
                        if i == 0:
                            reduced_h = h*w
                            reduced_w = 1
                        else:
                            reduced_h = math.ceil(reduced_h*reduce_ratio/100.)
                        
                    if i==0:
                        blk0 = nn.Sequential(*[ #GFnet
                            Block(
                                dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, init_values=init_values, h=reduced_h, w=reduced_w, dctGF_1d=dctGF_1d, GCM=GCM, reduce_ratio=-1, dftGF=dftGF, dctMLP=dctMLP,
                                drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[i], norm_layer=norm_layer, act_layer=act_layer, dctMSA=dctMSA, reduce_n=0, dctGF=dctGF, skip_connect=skip_connect)
                        ])
                    else:
                        blk0 = nn.Sequential(*[ #GFNet
                            Block(
                                dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio,qkv_bias=qkv_bias, init_values=init_values,h=reduced_h,w=reduced_w,dctGF_1d=dctGF_1d,GCM=GCM,reduce_ratio=reduce_ratio, dctMLP=dctMLP,
                                drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[i], norm_layer=norm_layer, act_layer=act_layer, dctMSA=dctMSA, dftGF=dftGF)
                        ])
                    self.blocks.append(blk0)
                    blk = nn.Sequential(*[ #ViT
                       Block(
                        dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, init_values=init_values, GCAM=GCAM, h=reduced_h, w=reduced_w, reduce_ratio_C=reduce_ratio_C, dctMLP=dctMLP,
                        drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[i], norm_layer=norm_layer, act_layer=act_layer, dctMSA=dctMSA)
                    for j in range(layers_in_stage-1)
                    ])
                    self.blocks.append(blk)
            else:
                if depth == 12: 
                    stage_num = 4
                    layers_in_stage = 3

                for i in range(stage_num):
                    if i == 0:
                        if L1st == True:
                            blk0 = nn.Sequential(*[
                            Block(
                                dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, init_values=init_values, h=h, w=w, dctGF_1d=dctGF_1d, GCM=GCM,
                                drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[i], norm_layer=norm_layer, act_layer=act_layer, dctMSA=dctMSA, reduce_n=0, dctGF=dctGF, skip_connect=skip_connect)
                            ])
                            self.blocks.append(blk0)
                            blk = nn.Sequential(*[
                               Block(
                                dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, init_values=init_values,
                                drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[i], norm_layer=norm_layer, act_layer=act_layer, dctMSA=dctMSA)
                            for j in range(layers_in_stage-1)
                            ])
                        else:
                            blk = nn.Sequential(*[
                                Block(
                                    dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, init_values=init_values,
                                    drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[i], norm_layer=norm_layer, act_layer=act_layer, dctMSA=dctMSA)
                            for j in range(layers_in_stage)
                            ])
                    else:
                        if i == 4: #depth==19, last stage
                            layers_in_stage -=1
                        if depth==13 and i==3:
                            layers_in_stage +=1

                        reduced_h = h-reduce_n*i
                        reduced_w = w-reduce_n*i

                        blk0 = nn.Sequential(*[
                            Block(
                                dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, init_values=init_values, h=reduced_h, w=reduced_w, dctGF_1d=dctGF_1d, GCM=GCM,
                                drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[i], norm_layer=norm_layer, act_layer=act_layer, dctMSA=dctMSA, reduce_n=reduce_n, dctGF=dctGF, skip_connect=skip_connect)
                        ])
                        self.blocks.append(blk0)
                        blk = nn.Sequential(*[
                           Block(
                            dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, init_values=init_values,
                            drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[i], norm_layer=norm_layer, act_layer=act_layer, dctMSA=dctMSA)
                        for j in range(layers_in_stage-1)
                        ])
                    self.blocks.append(blk)
        else:    
            self.blocks = nn.Sequential(*[
                block_fn(
                    dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, init_values=init_values, 
                    drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[i], norm_layer=norm_layer, act_layer=act_layer, dctMSA=dctMSA)
                for i in range(depth)])
        
        use_fc_norm = self.global_pool == 'avg'
        self.norm = norm_layer(embed_dim) if not use_fc_norm else nn.Identity()

        # Representation layer. Used for original ViT models w/ in21k pretraining.
        self.representation_size = representation_size
        self.pre_logits = nn.Identity()
        if representation_size:
            self._reset_representation(representation_size)

        # Classifier Head
        self.fc_norm = norm_layer(embed_dim) if use_fc_norm else nn.Identity()
        final_chs = self.representation_size if self.representation_size else self.embed_dim
        self.head = nn.Linear(final_chs, num_classes) if num_classes > 0 else nn.Identity()

        if weight_init != 'skip':
            self.init_weights(weight_init)

    def _reset_representation(self, representation_size):
        self.representation_size = representation_size
        if self.representation_size:
            self.pre_logits = nn.Sequential(OrderedDict([
                ('fc', nn.Linear(self.embed_dim, self.representation_size)),
                ('act', nn.Tanh())
            ]))
        else:
            self.pre_logits = nn.Identity()

    def init_weights(self, mode=''):
        assert mode in ('jax', 'jax_nlhb', 'moco', '')
        head_bias = -math.log(self.num_classes) if 'nlhb' in mode else 0.
        trunc_normal_(self.pos_embed, std=.02)
        if self.global_pool == 'token':
            nn.init.normal_(self.cls_token, std=1e-6)
        named_apply(get_init_weights_vit(mode, head_bias), self)

    def _init_weights(self, m):
        # this fn left here for compat with downstream users
        init_weights_vit_timm(m)

    @torch.jit.ignore()
    def load_pretrained(self, checkpoint_path, prefix=''):
        _load_weights(self, checkpoint_path, prefix)

    @torch.jit.ignore
    def no_weight_decay(self):
        return {'pos_embed', 'cls_token', 'dist_token'}

    @torch.jit.ignore
    def group_matcher(self, coarse=False):
        return dict(
            stem=r'^cls_token|pos_embed|patch_embed',  # stem and embed
            blocks=[(r'^blocks\.(\d+)', None), (r'^norm', (99999,))]
        )

    @torch.jit.ignore
    def set_grad_checkpointing(self, enable=True):
        self.grad_checkpointing = enable

    @torch.jit.ignore
    def get_classifier(self):
        return self.head

    def reset_classifier(self, num_classes: int, global_pool=None, representation_size=None):
        self.num_classes = num_classes
        if global_pool is not None:
            assert global_pool in ('', 'avg', 'token')
            self.global_pool = global_pool
        if representation_size is not None:
            self._reset_representation(representation_size)
        final_chs = self.representation_size if self.representation_size else self.embed_dim
        self.head = nn.Linear(final_chs, num_classes) if num_classes > 0 else nn.Identity()

    def forward_features(self, x):
        if self.YUV==True:
            x = rgb_to_yuv(x)
        x = self.patch_embed(x)
        if self.global_pool == 'token':
            x = torch.cat((self.cls_token.expand(x.shape[0], -1, -1), x), dim=1)
        x = self.pos_drop(x + self.pos_embed)
        if self.grad_checkpointing and not torch.jit.is_scripting():
            x = checkpoint_seq(self.blocks, x)
        else:
            # x = self.blocks(x)
            for blk in self.blocks:
                x = blk(x)
        x = self.norm(x)
        return x

    def forward_head(self, x, pre_logits: bool = False):
        if self.global_pool:
            x = x.mean(dim=1) if self.global_pool == 'avg' else x[:, 0]
            # x = x[:, 1:].mean(dim=1) if self.global_pool == 'avg' else x[:, 0]
        x = self.fc_norm(x)
        x = self.pre_logits(x)
        return x if pre_logits else self.head(x)

    def forward(self, x):
        x = self.forward_features(x)
        x = self.forward_head(x)
        return x


def init_weights_vit_timm(module: nn.Module, name: str = ''):
    """ ViT weight initialization, original timm impl (for reproducibility) """
    if isinstance(module, nn.Linear):
        trunc_normal_(module.weight, std=.02)
        if module.bias is not None:
            nn.init.zeros_(module.bias)


def init_weights_vit_jax(module: nn.Module, name: str = '', head_bias: float = 0.):
    """ ViT weight initialization, matching JAX (Flax) impl """
    if isinstance(module, nn.Linear):
        if name.startswith('head'):
            nn.init.zeros_(module.weight)
            nn.init.constant_(module.bias, head_bias)
        elif name.startswith('pre_logits'):
            lecun_normal_(module.weight)
            nn.init.zeros_(module.bias)
        else:
            nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                nn.init.normal_(module.bias, std=1e-6) if 'mlp' in name else nn.init.zeros_(module.bias)
    elif isinstance(module, nn.Conv2d):
        lecun_normal_(module.weight)
        if module.bias is not None:
            nn.init.zeros_(module.bias)


def init_weights_vit_moco(module: nn.Module, name: str = ''):
    """ ViT weight initialization, matching moco-v3 impl minus fixed PatchEmbed """
    if isinstance(module, nn.Linear):
        if 'qkv' in name:
            # treat the weights of Q, K, V separately
            val = math.sqrt(6. / float(module.weight.shape[0] // 3 + module.weight.shape[1]))
            nn.init.uniform_(module.weight, -val, val)
        else:
            nn.init.xavier_uniform_(module.weight)
        if module.bias is not None:
            nn.init.zeros_(module.bias)


def get_init_weights_vit(mode='jax', head_bias: float = 0.):
    if 'jax' in mode:
        return partial(init_weights_vit_jax, head_bias=head_bias)
    elif 'moco' in mode:
        return init_weights_vit_moco
    else:
        return init_weights_vit_timm


@torch.no_grad()
def _load_weights(model: VisionTransformer, checkpoint_path: str, prefix: str = ''):
    """ Load weights from .npz checkpoints for official Google Brain Flax implementation
    """
    import numpy as np

    def _n2p(w, t=True):
        if w.ndim == 4 and w.shape[0] == w.shape[1] == w.shape[2] == 1:
            w = w.flatten()
        if t:
            if w.ndim == 4:
                w = w.transpose([3, 2, 0, 1])
            elif w.ndim == 3:
                w = w.transpose([2, 0, 1])
            elif w.ndim == 2:
                w = w.transpose([1, 0])
        return torch.from_numpy(w)

    w = np.load(checkpoint_path)
    if not prefix and 'opt/target/embedding/kernel' in w:
        prefix = 'opt/target/'

    if hasattr(model.patch_embed, 'backbone'):
        # hybrid
        backbone = model.patch_embed.backbone
        stem_only = not hasattr(backbone, 'stem')
        stem = backbone if stem_only else backbone.stem
        stem.conv.weight.copy_(adapt_input_conv(stem.conv.weight.shape[1], _n2p(w[f'{prefix}conv_root/kernel'])))
        stem.norm.weight.copy_(_n2p(w[f'{prefix}gn_root/scale']))
        stem.norm.bias.copy_(_n2p(w[f'{prefix}gn_root/bias']))
        if not stem_only:
            for i, stage in enumerate(backbone.stages):
                for j, block in enumerate(stage.blocks):
                    bp = f'{prefix}block{i + 1}/unit{j + 1}/'
                    for r in range(3):
                        getattr(block, f'conv{r + 1}').weight.copy_(_n2p(w[f'{bp}conv{r + 1}/kernel']))
                        getattr(block, f'norm{r + 1}').weight.copy_(_n2p(w[f'{bp}gn{r + 1}/scale']))
                        getattr(block, f'norm{r + 1}').bias.copy_(_n2p(w[f'{bp}gn{r + 1}/bias']))
                    if block.downsample is not None:
                        block.downsample.conv.weight.copy_(_n2p(w[f'{bp}conv_proj/kernel']))
                        block.downsample.norm.weight.copy_(_n2p(w[f'{bp}gn_proj/scale']))
                        block.downsample.norm.bias.copy_(_n2p(w[f'{bp}gn_proj/bias']))
        embed_conv_w = _n2p(w[f'{prefix}embedding/kernel'])
    else:
        embed_conv_w = adapt_input_conv(
            model.patch_embed.proj.weight.shape[1], _n2p(w[f'{prefix}embedding/kernel']))
    model.patch_embed.proj.weight.copy_(embed_conv_w)
    model.patch_embed.proj.bias.copy_(_n2p(w[f'{prefix}embedding/bias']))
    if self.global_pool == 'token':
        model.cls_token.copy_(_n2p(w[f'{prefix}cls'], t=False))
    pos_embed_w = _n2p(w[f'{prefix}Transformer/posembed_input/pos_embedding'], t=False)
    if pos_embed_w.shape != model.pos_embed.shape:
        pos_embed_w = resize_pos_embed(  # resize pos embedding when different size from pretrained weights
            pos_embed_w, model.pos_embed, getattr(model, 'num_tokens', 1), model.patch_embed.grid_size)
    model.pos_embed.copy_(pos_embed_w)
    model.norm.weight.copy_(_n2p(w[f'{prefix}Transformer/encoder_norm/scale']))
    model.norm.bias.copy_(_n2p(w[f'{prefix}Transformer/encoder_norm/bias']))
    if isinstance(model.head, nn.Linear) and model.head.bias.shape[0] == w[f'{prefix}head/bias'].shape[-1]:
        model.head.weight.copy_(_n2p(w[f'{prefix}head/kernel']))
        model.head.bias.copy_(_n2p(w[f'{prefix}head/bias']))
    if isinstance(getattr(model.pre_logits, 'fc', None), nn.Linear) and f'{prefix}pre_logits/bias' in w:
        model.pre_logits.fc.weight.copy_(_n2p(w[f'{prefix}pre_logits/kernel']))
        model.pre_logits.fc.bias.copy_(_n2p(w[f'{prefix}pre_logits/bias']))
    for i, block in enumerate(model.blocks.children()):
        block_prefix = f'{prefix}Transformer/encoderblock_{i}/'
        mha_prefix = block_prefix + 'MultiHeadDotProductAttention_1/'
        block.norm1.weight.copy_(_n2p(w[f'{block_prefix}LayerNorm_0/scale']))
        block.norm1.bias.copy_(_n2p(w[f'{block_prefix}LayerNorm_0/bias']))
        block.attn.qkv.weight.copy_(torch.cat([
            _n2p(w[f'{mha_prefix}{n}/kernel'], t=False).flatten(1).T for n in ('query', 'key', 'value')]))
        block.attn.qkv.bias.copy_(torch.cat([
            _n2p(w[f'{mha_prefix}{n}/bias'], t=False).reshape(-1) for n in ('query', 'key', 'value')]))
        block.attn.proj.weight.copy_(_n2p(w[f'{mha_prefix}out/kernel']).flatten(1))
        block.attn.proj.bias.copy_(_n2p(w[f'{mha_prefix}out/bias']))
        for r in range(2):
            getattr(block.mlp, f'fc{r + 1}').weight.copy_(_n2p(w[f'{block_prefix}MlpBlock_3/Dense_{r}/kernel']))
            getattr(block.mlp, f'fc{r + 1}').bias.copy_(_n2p(w[f'{block_prefix}MlpBlock_3/Dense_{r}/bias']))
        block.norm2.weight.copy_(_n2p(w[f'{block_prefix}LayerNorm_2/scale']))
        block.norm2.bias.copy_(_n2p(w[f'{block_prefix}LayerNorm_2/bias']))


def resize_pos_embed(posemb, posemb_new, num_tokens=1, gs_new=()):
    # Rescale the grid of position embeddings when loading from state_dict. Adapted from
    # https://github.com/google-research/vision_transformer/blob/00883dd691c63a6830751563748663526e811cee/vit_jax/checkpoint.py#L224
    _logger.info('Resized position embedding: %s to %s', posemb.shape, posemb_new.shape)
    ntok_new = posemb_new.shape[1]
    if num_tokens:
        posemb_tok, posemb_grid = posemb[:, :num_tokens], posemb[0, num_tokens:]
        ntok_new -= num_tokens
    else:
        posemb_tok, posemb_grid = posemb[:, :0], posemb[0]
    gs_old = int(math.sqrt(len(posemb_grid)))
    if not len(gs_new):  # backwards compatibility
        gs_new = [int(math.sqrt(ntok_new))] * 2
    assert len(gs_new) >= 2
    _logger.info('Position embedding grid-size from %s to %s', [gs_old, gs_old], gs_new)
    posemb_grid = posemb_grid.reshape(1, gs_old, gs_old, -1).permute(0, 3, 1, 2)
    posemb_grid = F.interpolate(posemb_grid, size=gs_new, mode='bicubic', align_corners=False)
    posemb_grid = posemb_grid.permute(0, 2, 3, 1).reshape(1, gs_new[0] * gs_new[1], -1)
    posemb = torch.cat([posemb_tok, posemb_grid], dim=1)
    return posemb


def checkpoint_filter_fn(state_dict, model):
    """ convert patch embedding weight from manual patchify + linear proj to conv"""
    out_dict = {}
    if 'model' in state_dict:
        # For deit models
        state_dict = state_dict['model']
    for k, v in state_dict.items():
        if 'patch_embed.proj.weight' in k and len(v.shape) < 4:
            # For old models that I trained prior to conv based patchification
            O, I, H, W = model.patch_embed.proj.weight.shape
            v = v.reshape(O, -1, H, W)
        elif k == 'pos_embed' and v.shape != model.pos_embed.shape:
            # To resize pos embedding when using model at different size from pretrained weights
            v = resize_pos_embed(
                v, model.pos_embed, getattr(model, 'num_tokens', 1), model.patch_embed.grid_size)
        out_dict[k] = v
    return out_dict


def _create_vision_transformer(variant, pretrained=False, **kwargs):
    if kwargs.get('features_only', None):
        raise RuntimeError('features_only not implemented for Vision Transformer models.')

    # NOTE this extra code to support handling of repr size for in21k pretrained models
    pretrained_cfg = resolve_pretrained_cfg(variant, kwargs=kwargs)
    default_num_classes = pretrained_cfg['num_classes']
    num_classes = kwargs.get('num_classes', default_num_classes)
    repr_size = kwargs.pop('representation_size', None)
    if repr_size is not None and num_classes != default_num_classes:
        # Remove representation layer if fine-tuning. This may not always be the desired action,
        # but I feel better than doing nothing by default for fine-tuning. Perhaps a better interface?
        _logger.warning("Removing representation layer for fine-tuning.")
        repr_size = None
    
    model = build_model_with_cfg(
        VisionTransformer, variant, pretrained,
        pretrained_cfg=pretrained_cfg,
        representation_size=repr_size,
        pretrained_filter_fn=checkpoint_filter_fn,
        pretrained_custom_load='npz' in pretrained_cfg['url'],
        **kwargs)
    return model




def dct_fft_impl(v):
    return torch.view_as_real(torch.fft.fft(v, dim=1))

def idct_irfft_impl(V):
    return torch.fft.irfft(torch.view_as_complex(V), n=V.shape[1], dim=1)

def dct_1d(x, norm=None):
    """
    Discrete Cosine Transform, Type II (a.k.a. the DCT)
    For the meaning of the parameter `norm`, see:
    https://docs.scipy.org/doc/scipy-0.14.0/reference/generated/scipy.fftpack.dct.html
    :param x: the input signal
    :param norm: the normalization, None or 'ortho'
    :return: the DCT-II of the signal over the last dimension
    """
    x_shape = x.shape
    N = x_shape[-1]
    x = x.contiguous().view(-1, N)

    v = torch.cat([x[:, ::2], x[:, 1::2].flip([1])], dim=1)

    Vc = dct_fft_impl(v)

    k = - torch.arange(N, dtype=x.dtype, device=x.device)[None, :] * np.pi / (2 * N)
    W_r = torch.cos(k)
    W_i = torch.sin(k)

    V = Vc[:, :, 0] * W_r - Vc[:, :, 1] * W_i

    if norm == 'ortho':
        # V[:, 0] /= np.sqrt(N) * 2
        # V[:, 1:] /= np.sqrt(N / 2) * 2
        V[:, 0] /= torch.sqrt(N) * 2
        V[:, 1:] /= torch.sqrt(N / 2) * 2

    V = 2 * V.view(*x_shape)

    return V


def idct_1d(X, norm=None):
    """
    The inverse to DCT-II, which is a scaled Discrete Cosine Transform, Type III
    Our definition of idct is that idct(dct(x)) == x
    For the meaning of the parameter `norm`, see:
    https://docs.scipy.org/doc/scipy-0.14.0/reference/generated/scipy.fftpack.dct.html
    :param X: the input signal
    :param norm: the normalization, None or 'ortho'
    :return: the inverse DCT-II of the signal over the last dimension
    """

    x_shape = X.shape
    N = x_shape[-1]

    X_v = X.contiguous().view(-1, x_shape[-1]) / 2

    if norm == 'ortho':
        # X_v[:, 0] *= np.sqrt(N) * 2
        # X_v[:, 1:] *= np.sqrt(N / 2) * 2
        X_v[:, 0] *= torch.sqrt(N) * 2
        X_v[:, 1:] *= torch.sqrt(N / 2) * 2
    
    k = torch.arange(x_shape[-1], dtype=X.dtype, device=X.device)[None, :] * np.pi / (2 * N)
    W_r = torch.cos(k)
    W_i = torch.sin(k)

    V_t_r = X_v
    V_t_i = torch.cat([X_v[:, :1] * 0, -X_v.flip([1])[:, :-1]], dim=1)

    V_r = V_t_r * W_r - V_t_i * W_i
    V_i = V_t_r * W_i + V_t_i * W_r

    V = torch.cat([V_r.unsqueeze(2), V_i.unsqueeze(2)], dim=2)

    v = idct_irfft_impl(V)
    x = v.new_zeros(v.shape)
    x[:, ::2] += v[:, :N - (N // 2)]
    x[:, 1::2] += v.flip([1])[:, :N // 2]

    return x.view(*x_shape)

def qtable(h, w):
    Qtable_8x8 = torch.tensor([[16,	11,	10,	16,	24,	40,	51,	61],
              [12,	12,	14,	19,	26,	58,	60,	55],
              [14,	13,	16,	24,	40,	57,	69,	56],
              [14,	17,	22,	29,	51,	87,	80,	62],
              [18,	22,	37,	56,	68,	109,	103,	77],
              [24,	35,	55,	64,	81,	104,	113,	92],
              [49,	64,	78,	87,	103,	121,	120,	101],
              [72,	92,	95,	98,	112,	100,	103,	99]], dtype=torch.float32)
    
    Qtable_resized = F.interpolate(Qtable_8x8.unsqueeze(0).unsqueeze(0), size=(w, h), mode='bicubic', align_corners=False)
    Qtable_init = 1/Qtable_resized.squeeze(0).squeeze(0)
    
    return Qtable_init


def dct_2_1d(x, norm=None):
    """
    Discrete Cosine Transform, Type II (a.k.a. the DCT)
    For the meaning of the parameter `norm`, see:
    https://docs.scipy.org/doc/scipy-0.14.0/reference/generated/scipy.fftpack.dct.html
    :param x: the input signal
    :param norm: the normalization, None or 'ortho'
    :return: the DCT-II of the signal over the last dimension
    """
    x_shape = x.shape
    N = x_shape[-1]
    x = x.contiguous().view(-1, N)

    v = torch.cat([x[:, ::2], x[:, 1::2].flip([1])], dim=1)

    Vc = dct_fft_impl(v)

    k = - torch.arange(N, dtype=x.dtype, device=x.device)[None, :] * np.pi / (2 * N)
    W_r = torch.cos(k)
    W_i = torch.sin(k)

    V = Vc[:, :, 0] * W_r - Vc[:, :, 1] * W_i

    if norm == 'ortho':
        # V[:, 0] /= np.sqrt(N) * 2
        # V[:, 1:] /= np.sqrt(N / 2) * 2
        V[:, 0] /= torch.sqrt(N) * 2
        V[:, 1:] /= torch.sqrt(N / 2) * 2

    V = 2 * V.view(*x_shape)

    return V


def idct_2_1d(X, norm=None):
    """
    The inverse to DCT-II, which is a scaled Discrete Cosine Transform, Type III
    Our definition of idct is that idct(dct(x)) == x
    For the meaning of the parameter `norm`, see:
    https://docs.scipy.org/doc/scipy-0.14.0/reference/generated/scipy.fftpack.dct.html
    :param X: the input signal
    :param norm: the normalization, None or 'ortho'
    :return: the inverse DCT-II of the signal over the last dimension
    """

    x_shape = X.shape
    N = x_shape[-1]

    X_v = X.contiguous().view(-1, x_shape[-1]) / 2

    if norm == 'ortho':
        # X_v[:, 0] *= np.sqrt(N) * 2
        # X_v[:, 1:] *= np.sqrt(N / 2) * 2
        X_v[:, 0] *= torch.sqrt(N) * 2
        X_v[:, 1:] *= torch.sqrt(N / 2) * 2

    k = torch.arange(x_shape[-1], dtype=X.dtype, device=X.device)[None, :] * np.pi / (2 * N)
    W_r = torch.cos(k)
    W_i = torch.sin(k)

    V_t_r = X_v
    V_t_i = torch.cat([X_v[:, :1] * 0, -X_v.flip([1])[:, :-1]], dim=1)

    V_r = V_t_r * W_r - V_t_i * W_i
    V_i = V_t_r * W_i + V_t_i * W_r

    V = torch.cat([V_r.unsqueeze(2), V_i.unsqueeze(2)], dim=2)

    v = idct_irfft_impl(V)
    x = v.new_zeros(v.shape)
    x[:, ::2] += v[:, :N - (N // 2)]
    x[:, 1::2] += v.flip([1])[:, :N // 2]

    return x.view(*x_shape)
