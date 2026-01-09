from typing import Union, List, Tuple, Type

import torch
from torch import nn
from torch.nn.modules.conv import _ConvNd

from dynamic_network_architectures.building_blocks.helper import get_matching_instancenorm


class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding):
        super().__init__()
        self.norm = nn.InstanceNorm3d(out_channels)
        self.conv = nn.Conv3d(in_channels, out_channels, kernel_size, stride, padding)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.relu(self.conv(self.norm(x)))


class MixtureOfFeaturesDifference(nn.Module):
    def __init__(self, features_per_stage: Union[int, List[int], Tuple[int, ...]], conv_op: Type[_ConvNd], topk: int = 6):
        super().__init__()

        for conv in range(len(features_per_stage)):
            setattr(self, f"conv_{conv}", ConvBlock(features_per_stage[conv], features_per_stage[conv], 3, 1, 1))

        self.gap = nn.AdaptiveAvgPool3d((1, 1, 1))
        self.softmax = nn.Softmax(dim=1)
        self.topk = topk
    
    def forward(self, skips_s, skips_f1, skips_f2=None):
        if skips_f2 is None:
            for num_skip, (skip_s, skip_f1) in enumerate(zip(skips_s, skips_f1)):
                conv = getattr(self, f"conv_{num_skip}")
                skip_s = conv(skip_s)
                skip_f1 = conv(skip_f1)
                diff1 = conv(skip_s - skip_f1)

                feas = torch.stack([skip_s, skip_f1, diff1], dim=1)
                probs = self.softmax(self.gap(feas))

                if self.topk >= 3:
                    weights = probs
                elif self.topk == 0:
                    # average
                    weights = torch.ones_like(probs) / probs.shape[1]
                else:
                    weights, topk_idx = torch.topk(probs, k=self.topk, dim=1)
                    weights = weights / weights.sum(dim=1, keepdim=True)
                    
                    full_weights = torch.zeros_like(probs)
                    full_weights.scatter_(1, topk_idx, weights)
                    weights = full_weights
                
                aggr = (weights * feas).sum(dim=1)
                skips_s[num_skip] = aggr
            return skips_s
        
        elif skips_f2 is not None:
            for num_skip, (skip_s, skip_f1, skip_f2) in enumerate(zip(skips_s, skips_f1, skips_f2)):
                conv = getattr(self, f"conv_{num_skip}")
                skip_s = conv(skip_s)
                skip_f1 = conv(skip_f1)
                skip_f2 = conv(skip_f2)
                diff1 = conv(skip_s - skip_f1)
                diff2 = conv(skip_s - skip_f2)
                diff3 = conv(skip_f1 - skip_f2)

                feas = torch.stack([skip_s, skip_f1, skip_f2, diff1, diff2, diff3], dim=1)
                probs = self.softmax(self.gap(feas))

                if self.topk >= 6:
                    weights = probs
                elif self.topk == 0:
                    # average
                    weights = torch.ones_like(probs) / probs.shape[1]
                else:
                    weights, topk_idx = torch.topk(probs, k=self.topk, dim=1)
                    weights = weights / weights.sum(dim=1, keepdim=True)

                    full_weights = torch.zeros_like(probs)
                    full_weights.scatter_(1, topk_idx, weights)
                    weights = full_weights
                aggr = (weights * feas).sum(dim=1)
                skips_s[num_skip] = aggr
            return skips_s
