# AUTOGENERATED! DO NOT EDIT! File to edit: ../nbs/01_fpn.ipynb.

# %% auto 0
__all__ = ['FPN']

# %% ../nbs/01_fpn.ipynb 2
import copy
import timm
import torch.nn as nn

# %% ../nbs/01_fpn.ipynb 3
class FPN(nn.Module):
    """implementation of FPN (feature pyramid) https://arxiv.org/pdf/1612.03144.pdf"""
    def __init__(self, feature_channels, out_indices):
        super().__init__()
        
        self.feature_channels=feature_channels
        self.out_indices=out_indices
        self.up_sample=nn.Upsample(scale_factor=2)
        self.lat_convs=self._get_lat_convs()
    
    #TODO add conv3x3 after merge to reduce aliasing effect
    def conv1x1(self, in_c, out_c):
        return nn.Conv2d(in_c, out_c, kernel_size=1)
        
    def _get_lat_convs(self):
        """generate 1x1convs for lateral connections"""
        feature_channels=copy.deepcopy(self.feature_channels)
        feature_channels.insert(0, feature_channels[0])
        assert len(feature_channels)==len(self.out_indices)+1 
        lat_convs=[]
        for idx in range(0, len(feature_channels)-1, 1): 
            in_c, out_c = feature_channels[idx+1], 256
            conv1=self.conv1x1(in_c, out_c)
            lat_convs.append(conv1)
        return nn.Sequential(*lat_convs)
    
    def forward(self, x):
        lat_feats=[]
        for idx in range(len(x)-1, -1, -1):
            if idx==3: merge=self.lat_convs[idx](x[idx])
            else: merge=self.up_sample(merge)+self.lat_convs[idx](x[idx])
            lat_feats.append(merge)
        return lat_feats
