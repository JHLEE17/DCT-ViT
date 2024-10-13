""" Image to Patch Embedding using Conv2d

A convolution based approach to patchifying a 2D image w/ embedding projection.

Based on the impl in https://github.com/google-research/vision_transformer

Hacked together by / Copyright 2020 Ross Wightman
"""
from torch import nn as nn
import torchjpeg.dct as dct
import torch.nn.functional as F
from .helpers import to_2tuple
from .trace_utils import _assert

class PatchEmbedDCT_P(nn.Module):
    """ 2D Image to Patch Embedding
    """
    def __init__(self, img_size=224, patch_size=16, in_chans=3, embed_dim=768, norm_layer=None, flatten=True, DCT_P_PE=8, model_input_size=224, ro_ds=1):
        super().__init__()
        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)
        model_input_size = to_2tuple(model_input_size)
        self.img_size = img_size
        self.patch_size = patch_size
        self.model_input_size = model_input_size
        self.ro_ds = ro_ds
        
        self.grid_size = (model_input_size[0] // patch_size[0], model_input_size[1] // patch_size[1])
        self.num_patches = self.grid_size[0] * self.grid_size[1]
        self.flatten = flatten       
        if ro_ds > 1:
            # print('ro_ds: ', self.ro_ds)
            self.proj = nn.Conv2d(48, embed_dim, kernel_size=1)
        else:
            self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)
        
        self.norm = norm_layer(embed_dim) if norm_layer else nn.Identity()
        
        self.dct_block_size = DCT_P_PE
        self.embed_dim = embed_dim
        

    def forward(self, x):
        
        B, C, H, W = x.shape
        # _assert(H == self.img_size[0], f"Input image height ({H}) doesn't match model ({self.img_size[0]}).")
        # _assert(W == self.img_size[1], f"Input image width ({W}) doesn't match model ({self.img_size[1]}).")
        
        if self.img_size == self.model_input_size:
            x = dct.deblockify(dct.block_dct(dct.blockify(x,self.dct_block_size)), self.model_input_size)    
        else:
            remain = int(self.dct_block_size/(self.img_size[0]/self.model_input_size[0])) # 8/2 = 4
            x = dct.deblockify(dct.block_dct(dct.blockify(x,self.dct_block_size))[:,:,:,:remain,:remain],self.model_input_size)    
            # x = dct.deblockify(dct.block_dct(dct.blockify(x,self.dct_block_size))[:,:,:,remain:,remain:],self.model_input_size)
        
        if self.ro_ds > 1:
            B, C, H, W = x.shape
            x = x.view(B, C, H // self.ro_ds, self.ro_ds, W // self.ro_ds, self.ro_ds)  # (B, C, H//bs, bs, W//bs, bs)
            x = x.permute(0, 3, 5, 1, 2, 4).contiguous()  # (B, bs, bs, C, H//bs, W//bs)
            x = x.view(B, C * (self.ro_ds ** 2), H // self.ro_ds, W // self.ro_ds)  # (B, C*bs^2, H//bs, W//bs)
            
        x = self.proj(x)
        if self.flatten:
            x = x.flatten(2).transpose(1, 2)  # BCHW -> BNC
            
        #             BCNww  ->   BCNW ->     BNCW     -> BND
        # x = dct.blockify(x,16).flatten(3).transpose(1,2).flatten(2)
        # x = F.interpolate(x, size=self.embed_dim, mode='linear',align_corners=False)
        
        x = self.norm(x)
        return x
