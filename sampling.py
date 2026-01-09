import argparse, os, sys, datetime, glob, importlib
from omegaconf import OmegaConf
import nibabel as nib
import numpy as np
import torch
from pytorch_lightning.trainer import Trainer

import warnings
warnings.filterwarnings("ignore")

from main import instantiate_from_config


def get_obj_from_str(string, reload=False):
    module, cls = string.rsplit(".", 1)
    if reload:
        module_imp = importlib.import_module(module)
        importlib.reload(module_imp)
    return getattr(importlib.import_module(module, package=None), cls)


def get_parser(**parser_kwargs):
    def str2bool(v):
        if isinstance(v, bool):
            return v
        if v.lower() in ("yes", "true", "t", "y", "1"):
            return True
        elif v.lower() in ("no", "false", "f", "n", "0"):
            return False
        else:
            raise argparse.ArgumentTypeError("Boolean value expected.")

    parser = argparse.ArgumentParser(**parser_kwargs)
    parser.add_argument(
        "-n",
        "--name",
        type=str,
        const=True,
        default="",
        nargs="?",
        help="postfix for logdir",
    )
    parser.add_argument(
        "-r",
        "--resume",
        type=str,
        const=True,
        default="",
        nargs="?",
        help="resume from logdir or checkpoint in logdir",
    )
    parser.add_argument(
        "-b",
        "--base",
        nargs="*",
        metavar="base_config.yaml",
        help="paths to base configs. Loaded from left-to-right. "
        "Parameters can be overwritten or added with command-line options of the form `--key value`.",
        default=list(),
    )
    parser.add_argument(
        "-t",
        "--train",
        type=str2bool,
        const=True,
        default=False,
        nargs="?",
        help="train",
    )
    parser.add_argument(
        "--no-test",
        type=str2bool,
        const=True,
        default=False,
        nargs="?",
        help="disable test",
    )
    parser.add_argument("-p", "--project", help="name of new or path to existing project")
    parser.add_argument(
        "-d",
        "--debug",
        type=str2bool,
        nargs="?",
        const=True,
        default=False,
        help="enable post-mortem debugging",
    )
    parser.add_argument(
        "-s",
        "--seed",
        type=int,
        default=23,
        help="seed for seed_everything",
    )
    parser.add_argument(
        "-f",
        "--postfix",
        type=str,
        default="",
        help="post-postfix for default name",
    )

    return parser

def get_affine(src_path):
    M = nib.load(src_path).affine[:3, :3]
    P = np.zeros_like(M)
    max_abs_indices = np.argmax(np.abs(M), axis=1)
    for i, col_idx in enumerate(max_abs_indices):
        P[i, col_idx] = np.sign(M[i, col_idx])
    
    P_inv = np.linalg.inv(P)
    new_M = M @ P_inv
    affine = np.eye(4)
    affine[:3, :3] = new_M
         
    return affine

def save_nii(img, name, path, src_path):
    img = np.transpose(img, (1, 2, 3, 0))

    affine = get_affine(src_path)
    nifti_img = nib.Nifti1Image(img, affine)
    nib.save(nifti_img, f"{path}/{name}.nii.gz")

def maybe_mkdir(path):
    if not os.path.exists(path):
        os.makedirs(path, exist_ok=True)

if __name__ == "__main__":
    parser = get_parser()
    parser = Trainer.add_argparse_args(parser)
    opt, unknown = parser.parse_known_args()

    configs = [OmegaConf.load(cfg) for cfg in opt.base]
    cli = OmegaConf.from_dotlist(unknown)
    config = OmegaConf.merge(*configs, cli)
    
    # model
    model = instantiate_from_config(config.model)
    model.eval()
    model.to("cuda")

    # data
    data = instantiate_from_config(config.data)
    data.prepare_data()
    data.setup()

    test_loader = data.test_dataloader()
    root_path = "results/" + opt.base[0].split("/")[-1].split(".")[0]
    for batch in test_loader:
        src_path = batch["path"][0]
        subject = batch["subject_id"][0]
        print("processing...", subject)

        sub_path = os.path.join(root_path, subject)
        maybe_mkdir(sub_path)
        sample_list = []

        batch_on_device = {}
        for k, v in batch.items():
            if isinstance(v, torch.Tensor):
                batch_on_device[k] = v.to(model.device)
            else:
                batch_on_device[k] = v

        batch = batch_on_device

        log = model.log_images(batch, sample_steps=4)
        recon = log["reconstructions"][0].detach().cpu().numpy()
        sample_list.append(recon)
        save_nii(recon, "pred", sub_path, src_path)

        t1 = batch["t1"][0].detach().cpu().numpy()
        t2 = batch["t2"][0].detach().cpu().numpy()
        flair = batch["flair"][0].detach().cpu().numpy()
        target = batch["t1ce"][0].detach().cpu().numpy()
        save_nii(t1, "t1", sub_path, src_path)
        save_nii(t2, "t2", sub_path, src_path)
        save_nii(flair, "flair", sub_path, src_path)
        save_nii(target, "ground_truth", sub_path, src_path)
