import torch
from torch.utils.data import DataLoader
import getpass
username = getpass.getuser()
import torchvision.transforms as transforms
from loguru import logger as log
import os
import torchvision
import random
from prefetch_generator import BackgroundGenerator
import torch, os
from PIL import Image
import pandas as pd
import numpy as np
import torch.nn.functional as F
from torch.utils.data import Dataset
import torch
import json


def get_augmentation(aug_type, res_size=256, crop_size=224):
    transform_list = [] # 初始化
    if aug_type == "moco-v2":
        transform_list = [
            transforms.RandomResizedCrop(crop_size, scale=(0.2, 1.0)),
            transforms.RandomApply(
                [transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)],
                p=0.8,
            ),
            transforms.RandomGrayscale(p=0.2),
            transforms.RandomApply([GaussianBlur([0.1, 2.0])], p=0.5), # 注意：你需要确保定义了 GaussianBlur 类
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor()
        ]
    elif aug_type == "moco-v2-light":
        transform_list = [
            transforms.Resize(res_size),
            transforms.RandomApply(
                [transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)],
                p=0.8,
            ),
            transforms.RandomGrayscale(p=0.2),
            transforms.RandomApply([GaussianBlur([0.1, 2.0])], p=0.5),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor()
        ]
    elif aug_type == "moco-v1":
        transform_list = [
            transforms.RandomResizedCrop(crop_size, scale=(0.2, 1.0)),
            transforms.RandomGrayscale(p=0.2),
            transforms.ColorJitter(0.4, 0.4, 0.4, 0.4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor()
        ]
    elif aug_type == "plain":
        transform_list = [
            transforms.Resize(res_size),
            transforms.RandomCrop(crop_size),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor()
        ]
    elif aug_type == "clip_inference":
        transform_list = [
            transforms.Resize(crop_size, interpolation=Image.BICUBIC),
            transforms.CenterCrop(crop_size),
            transforms.ToTensor()
        ]
    elif aug_type == "test":
        transform_list = [
            transforms.Resize(res_size),
            transforms.CenterCrop(crop_size),
            transforms.ToTensor()
        ]
    else:
        return None
    
    return transforms.Compose(transform_list)
        
def complete_data_dir_path(root, dataset_name):
    mapping = {
        "imagenet": "ImageNet",
        "imagenet_c": "ImageNet",
        "imagenet_3dcc": "ImageNet",
        "imagenet_k": os.path.join("ImageNet-Sketch", "sketch"),
        "imagenet_r": os.path.join("ImageNet-R", "imagenet_r"),
        "imagenet_c_bar": "ImageNet",
    }
    return os.path.join(root, mapping[dataset_name])

def build_imagenet_k_r_v2(cfg):
    transform = get_augmentation(aug_type="test", res_size=(256, 256), crop_size=224)
    data_dir = complete_data_dir_path(root=cfg.DATA_DIR, dataset_name=cfg.CORRUPTION.DATASET)
    log.info(f"==>> data_dir:  {data_dir}")
    test_dataset = torchvision.datasets.ImageFolder(root=data_dir, transform=transform)
    test_loader = DataLoader(
        test_dataset,
        batch_size=cfg.TEST.BATCH_SIZE,
        shuffle=True,
        num_workers=4,
        pin_memory=False,
    )
    return test_loader