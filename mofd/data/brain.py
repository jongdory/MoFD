import os
import numpy as np
import pandas

import torch
from torch.utils.data import Dataset

from monai import transforms
from monai.data import Dataset as MonaiDataset


def get_transforms(phase="train"):
    modalities = ["t1", "t1ce", "t2", "flair"]

    if phase == "train":
        train_transforms = transforms.Compose(
            [   
                transforms.RandFlipd(keys=modalities, prob=0.1, spatial_axis=0, allow_missing_keys=True),
                transforms.RandFlipd(keys=modalities, prob=0.1, spatial_axis=1, allow_missing_keys=True),
                transforms.RandFlipd(keys=modalities, prob=0.1, spatial_axis=2, allow_missing_keys=True),
            ]
        )
    
    return transforms.Compose(
        [
            transforms.LoadImaged(keys=modalities, allow_missing_keys=True),
            transforms.AddChanneld(keys=modalities, allow_missing_keys=True),
            transforms.Orientationd(keys=modalities, axcodes="RAS", allow_missing_keys=True),
            transforms.EnsureTyped(keys=modalities, allow_missing_keys=True),
            transforms.CropForegroundd(keys=modalities, source_key="t1", margin=0, allow_missing_keys=True),
            transforms.SpatialPadd(keys=modalities, spatial_size=(192, 192, 96), allow_missing_keys=True), # 144, 192, 144
            transforms.CenterSpatialCropd(keys=modalities, roi_size=(192, 192, 96), allow_missing_keys=True),
            transforms.ScaleIntensityRangePercentilesd(keys=modalities, lower=0, upper=99.5, b_min=-1, b_max=1, allow_missing_keys=True),
            train_transforms if phase == "train" else transforms.Compose([])
        ]
    )


def get_upenn_dataset(data_path, csv_path=None, phase="train"):
    transform = get_transforms(phase=phase)
    
    datalist = []
    
    df = pandas.read_csv(csv_path)
    for sub_id in df["id"].tolist():
        split_list = df[df["id"] == sub_id]["split"].tolist()
        if split_list and split_list[0] == phase:
            datalist.append(sub_id)
    

    data = []
    for subject in datalist:
        sub_path = os.path.join(data_path, subject)
        
        if os.path.exists(sub_path) == False: 
            continue

        t1 = os.path.join(sub_path, f"{subject}_T1.nii.gz")
        t1ce = os.path.join(sub_path, f"{subject}_T1GD.nii.gz")
        t2 = os.path.join(sub_path, f"{subject}_T2.nii.gz")
        flair = os.path.join(sub_path, f"{subject}_FLAIR.nii.gz")

        if os.path.exists(t1) == False or os.path.exists(t1ce) == False or os.path.exists(t2) == False or os.path.exists(flair) == False:
            continue

        data.append({"t1":t1, "t1ce":t1ce, "t2":t2, "flair":flair, "subject_id": subject, "path": t1})        
                    
    print(phase, " num of subject:", len(data))

    return MonaiDataset(data=data, transform=transform)


def get_ucsf_dataset(data_path, csv_path=None, phase="train"):
    transform = get_transforms(phase=phase)
    datalist = []
    
    df = pandas.read_csv(csv_path)
    for sub_id in df["id"].tolist():
        split_list = df[df["id"] == sub_id]["split"].tolist()
        if split_list and split_list[0] == phase:
            datalist.append(sub_id)

    data = []
    for subject in os.listdir(data_path):
        sub_path = os.path.join(data_path, subject)
        subject = subject.split("_")[0]
        if subject not in datalist:
            continue
        if os.path.exists(sub_path) == False: 
            continue

        t1 = os.path.join(sub_path, f"{subject}_T1_bias.nii.gz")
        t1ce = os.path.join(sub_path, f"{subject}_T1c_bias.nii.gz")
        t2 = os.path.join(sub_path, f"{subject}_T2_bias.nii.gz")
        flair = os.path.join(sub_path, f"{subject}_FLAIR_bias.nii.gz")

        if os.path.exists(t1) == False or os.path.exists(t1ce) == False or os.path.exists(t2) == False or os.path.exists(flair) == False:
            continue

        data.append({"t1":t1, "t1ce":t1ce, "t2":t2, "flair":flair, "subject_id": subject, "path": t1})        
                    
    print(phase, " num of subject:", len(data))

    return MonaiDataset(data=data, transform=transform)

class Brain3DBase(Dataset):
    def __init__(self, **kwargs):
        super().__init__()
        self.data = None

    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, i):
        return self.data[i]


class Brain3DTrain(Brain3DBase):
    def __init__(self, phase="train"):
        self.data = UPennTrain(data_path="data/UPENN-GBM/", 
                                csv_path="data/UPENN-GBM/UPENN-GBM_split.csv",
                                phase=phase) + \
                    UCSFTrain(data_path="data/UCSF-PDGM/", 
                                csv_path="data/UCSF-PDGM/UCSF-PDGM_split.csv",
                                phase=phase)

class Brain3DTest(Brain3DBase):
    def __init__(self, phase="val"):
        self.data = UPennTest(data_path="data/UPENN-GBM/",
                                csv_path="data/UPENN-GBM/UPENN-GBM_split.csv",
                                phase=phase) + \
                    UCSFTest(data_path="data/UCSF-PDGM/", 
                                csv_path="data/UCSF-PDGM/UCSF-PDGM_split.csv",
                                phase=phase)
    
    
# T1, T2 -> FLAIR
class Base(Brain3DBase):
    def __init__(self):
        self.modalities = ["t1", "t1ce", "t2", "flair"]
        
    def __len__(self):
        return len(self.data)

    def __getitem__(self, i):
        item = dict(self.data[i])

        return item
    


class UPennTrain(Base):
    def __init__(self, data_path, csv_path=None, phase="train"):
        super().__init__()
        self.data = get_upenn_dataset(data_path, csv_path, phase)

class UPennTest(Base):
    def __init__(self, data_path, csv_path=None, phase="test"):
        super().__init__()
        self.data = get_upenn_dataset(data_path, csv_path, phase)

class UCSFTrain(Base):
    def __init__(self, data_path, csv_path=None, phase="train"):
        super().__init__()
        self.data = get_ucsf_dataset(data_path, csv_path, phase)
    
class UCSFTest(Base):
    def __init__(self, data_path, csv_path=None, phase="test"):
        super().__init__()
        self.data = get_ucsf_dataset(data_path, csv_path, phase)