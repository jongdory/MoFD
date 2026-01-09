import pydoc
import warnings
from os.path import join
from torch import nn
import torch
import torch.nn.functional as F
from math import exp

import dynamic_network_architectures

from mofd.models.utils import recursive_find_python_class
from mofd.models.building_blocks.mofd import MixtureOfFeaturesDifference

from einops import rearrange


class MOFDUNet(nn.Module):
    def __init__(self, input_channels, num_classes, backbone_class_name, conv_op=nn.Conv3d, norm_op=nn.InstanceNorm3d, nonlin=nn.LeakyReLU, topk=6, **architecture_kwargs):
        super().__init__()
        backbone_class = pydoc.locate(backbone_class_name)
        # sometimes things move around, this makes it so that we can at least recover some of that
        if backbone_class is None:
            warnings.warn(f'Network class {backbone_class_name} not found. Attempting to locate it within '
                        f'dynamic_network_architectures.architectures...')
            backbone_class = recursive_find_python_class(join(dynamic_network_architectures.__path__[0], "architectures"),
                                                backbone_class_name.split(".")[-1],
                                                'dynamic_network_architectures.architectures')
            if backbone_class is not None:
                print(f'FOUND IT: {backbone_class}')
            else:
                raise ImportError('Network class could not be found, please check/correct your plans file')

        # basic channel concatenation
        self.backbone = backbone_class(
            input_channels=input_channels,
            num_classes=num_classes,
            conv_op=conv_op,
            norm_op=norm_op,
            nonlin=nonlin,
            **architecture_kwargs
        )

        self.skip_diff_weighting = MixtureOfFeaturesDifference(architecture_kwargs['features_per_stage'], conv_op, topk)
        self.loss = AlignLoss()

    def forward(self, d_s, d_f1, d_f2=None):
        skips_s = self.backbone.encoder(d_s)
        skips_f1 = self.backbone.encoder(d_f1)

        if d_f2 is not None:
            skips_f2 = self.backbone.encoder(d_f2)
            loss_al = self.loss_align(skips_s[-1], skips_f1[-1], skips_f2[-1])
        else:
            skips_f2 = None
            loss_al = self.loss_align(skips_s[-1], skips_f1[-1])

        skips = self.skip_diff_weighting(skips_s, skips_f1, skips_f2)
        x = self.backbone.decoder(skips)
        
        return x, loss_al

    def loss_align(self, s, f1, f2=None):
        if f2 is None:
            return self.loss(s, f1)
        else:
            return (self.loss(s, f1) + self.loss(s, f2) + self.loss(f1, f2)) / 3.0
    
    def __getattr__(self, name):
        try:
            return super().__getattr__(name)
        except AttributeError:
            if hasattr(self.backbone, name):
                return getattr(self.backbone, name)
            raise

    def __setattr__(self, name, value):
        if name != 'backbone' and hasattr(self, 'backbone') and hasattr(self.backbone, name):
            setattr(self.backbone, name, value)
        else:
            super().__setattr__(name, value)


def gaussian(window_size, sigma):
    gauss = torch.Tensor([exp(-(x - window_size // 2) ** 2 / float(2 * sigma ** 2)) for x in range(window_size)])
    return gauss / gauss.sum()

def create_window(window_size, channel):
    _1D_window = gaussian(window_size, 1.5).unsqueeze(1)
    _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
    window = _2D_window.expand(channel, 1, window_size, window_size).contiguous()
    return window

def structural_similarity(img1, img2, window_size=11, window=None, size_average=True):
    img1 = rearrange(img1, "b c h w d -> b (c d) h w")
    img2 = rearrange(img2, "b c h w d -> b (c d) h w")
    (_, channel, height, width) = img1.size()
    
    if window is None or window.size(0) != channel:
        real_size = min(window_size, height, width)
        window = create_window(real_size, channel=channel).to(img1.device).type(img1.dtype)
    
    sigma12 = F.conv2d(img1 * img2, window, padding=0, groups=channel) - (
        F.conv2d(img1, window, padding=0, groups=channel) * F.conv2d(img2, window, padding=0, groups=channel))
    sigma1_sq = F.conv2d(img1 * img1, window, padding=0, groups=channel) - F.conv2d(img1, window, padding=0, groups=channel) ** 2
    sigma2_sq = F.conv2d(img2 * img2, window, padding=0, groups=channel) - F.conv2d(img2, window, padding=0, groups=channel) ** 2
    sigma1_sq = torch.clamp(sigma1_sq, min=1e-6)
    sigma2_sq = torch.clamp(sigma2_sq, min=1e-6)

    C2 = 1e-4  
    structural_map = sigma12 / (torch.sqrt(sigma1_sq) * torch.sqrt(sigma2_sq) + C2)
    
    if size_average:
        return structural_map.mean()
    else:
        return structural_map.mean(dim=[1, 2, 3])

class AlignLoss(torch.nn.Module):
    def __init__(self, window_size=11, size_average=True):
        super(AlignLoss, self).__init__()
        self.window_size = window_size
        self.size_average = size_average
        self.window = None
        self.channel = None

    def forward(self, img1, img2):
        (_, channel, _, _, _) = img1.size()
        if self.window is None or self.channel != channel:
            self.window = create_window(self.window_size, channel).to(img1.device).type(img1.dtype)
            self.channel = channel
        return 1 - structural_similarity(img1, img2, window=self.window, window_size=self.window_size, size_average=self.size_average)