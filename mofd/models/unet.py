import numpy as np
import random
import torch
import math
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
from einops import rearrange
from functools import partial

from main import instantiate_from_config
from mofd.models.architectures.unet_diff import UNetDiff

def stopgrad(x):
    return x.detach()


def adaptive_l2_loss(error, gamma=0.5, c=1e-3):
    """
    Adaptive L2 loss: sg(w) * ||Δ||_2^2, where w = 1 / (||Δ||^2 + c)^p, p = 1 - γ
    Args:
        error: Tensor of shape (B, C, W, H, D)
        gamma: Power used in original ||Δ||^{2γ} loss
        c: Small constant for stability
    Returns:
        Scalar loss
    """
    delta_sq = torch.mean(error ** 2, dim=(1, 2, 3, 4), keepdim=False)
    p = 1.0 - gamma
    w = 1.0 / (delta_sq + c).pow(p)
    loss = delta_sq  # ||Δ||^2
    return (stopgrad(w) * loss).mean()


class UNet(pl.LightningModule):
    def __init__(self,
                 ckpt_path=None,
                 ignore_keys=[],
                 backbone_config=None,
                 lossconfig=None,
                 source=["t1", "t2", "flair"],
                 target="t1ce",
                 mul = 1,
                 ):
        super().__init__()
        
        self.backbone = instantiate_from_config(backbone_config)
        self.loss = instantiate_from_config(lossconfig)
        self.source = source
        self.target = target

        if ckpt_path is not None:
            self.init_from_ckpt(ckpt_path, ignore_keys=ignore_keys)      

    def init_from_ckpt(self, path, ignore_keys=list()):
        sd = torch.load(path, map_location="cpu")["state_dict"]
        keys = list(sd.keys())
        for k in keys:
            for ik in ignore_keys:
                if k.startswith(ik):
                    print("Deleting key {} from state_dict.".format(k))
                    del sd[k]
        self.load_state_dict(sd, strict=False)
        print(f"Restored from {path}")
    
    def forward(self, d_s, d_f1=None, d_f2=None):
        xrec, loss = self.backbone(d_s, d_f1, d_f2)

        return xrec, loss

    def get_input(self, batch, k):
        x = batch[k]
        return x.float()

    def calculate_psnr(self, pred, target):
        mse = torch.mean((pred - target) ** 2)
        psnr = 20 * torch.log10(1 / (mse + 1e-6))
        return psnr

    def training_step(self, batch, batch_idx, optimizer_idx=0):
        d_s = self.get_input(batch, self.source[0])
        d_f1 = None 
        d_f2 = None

        if len(self.source) == 2:
            d_f1 = self.get_input(batch, self.source[1])
        elif len(self.source) == 3:
            d_f1 = self.get_input(batch, self.source[1])
            d_f2 = self.get_input(batch, self.source[2])

        x = self.get_input(batch, self.target)

        xrec, loss_align = self(d_s, d_f1, d_f2)
        if optimizer_idx == 0:
            # autoencode
            aeloss, log_dict_ae = self.loss(loss_align, x, xrec, optimizer_idx, self.global_step,
                                            last_layer=self.get_last_layer(), split="train")

            self.log("train/aeloss", aeloss, prog_bar=True, logger=True, on_step=True, on_epoch=True)
            self.log_dict(log_dict_ae, prog_bar=False, logger=True, on_step=True, on_epoch=True)
            return aeloss

        if optimizer_idx == 1:
            # discriminator
            discloss, log_dict_disc = self.loss(loss_align, x, xrec, optimizer_idx, self.global_step,
                                            last_layer=self.get_last_layer(), split="train")
            self.log("train/discloss", discloss, prog_bar=True, logger=True, on_step=True, on_epoch=True)
            self.log_dict(log_dict_disc, prog_bar=False, logger=True, on_step=True, on_epoch=True)
            return discloss

    def validation_step(self, batch, batch_idx):
        d_s = self.get_input(batch, self.source[0])
        d_f1 = None
        d_f2 = None
        if len(self.source) == 2:
            d_f1 = self.get_input(batch, self.source[1])
        elif len(self.source) == 3:
            d_f1 = self.get_input(batch, self.source[1])
            d_f2 = self.get_input(batch, self.source[2])
        x = self.get_input(batch, self.target)

        xrec, loss_align = self(d_s, d_f1, d_f2)
        aeloss, log_dict_ae = self.loss(loss_align, x, xrec, 0, self.global_step,
                                            last_layer=self.get_last_layer(), split="val")

        discloss, log_dict_disc = self.loss(loss_align, x, xrec, 1, self.global_step,
                                            last_layer=self.get_last_layer(), split="val")

        self.log("val/aeloss", aeloss,
                   prog_bar=True, logger=True, on_step=True, on_epoch=True, sync_dist=True)

        psnr = self.calculate_psnr(xrec, x)
        self.log("val_psnr", psnr, prog_bar=True, logger=True, on_step=True, on_epoch=True, sync_dist=True)
        self.log_dict(log_dict_ae)
        self.log_dict(log_dict_disc)
        return self.log_dict

    def configure_optimizers(self):
        lr = self.learning_rate
    
        opt_ae = torch.optim.Adam(list(self.backbone.parameters()),
                                lr=lr, betas=(0.5, 0.9))

        opt_disc = torch.optim.Adam(self.loss.discriminator.parameters(),
                                    lr=lr, betas=(0.5, 0.9))

        return [opt_ae, opt_disc], []

    def get_last_layer(self):
        return self.backbone.decoder.seg_layers[-1].weight

    @torch.no_grad()
    def log_images(self, batch,  **kwargs):
        log = dict()
        
        d_s = self.get_input(batch, self.source[0])
        d_f1 = None
        d_f2 = None
        if len(self.source) == 2:
            d_f1 = self.get_input(batch, self.source[1])
        elif len(self.source) == 3:
            d_f1 = self.get_input(batch, self.source[1])
            d_f2 = self.get_input(batch, self.source[2])
        x = self.get_input(batch, self.target)
        
        xrec, _ = self(d_s, d_f1, d_f2)
        log["reconstructions"] = xrec
        log[self.source[0]] = d_s
        if len(self.source) >= 2:
            log[self.source[1]] = d_f1
        if len(self.source) >= 3:
            log[self.source[2]] = d_f2
        log["targets"] = x
        return log