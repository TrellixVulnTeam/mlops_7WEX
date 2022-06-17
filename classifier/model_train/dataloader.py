import glob
import random
import os
import numpy as np
import pandas as pd

from torch.utils.data import Dataset
from PIL import Image
import torchvision.transforms as transforms

import torch
import torchvision
import albumentations as A
import albumentations.pytorch
import cv2
from torch.utils.data import Dataset, DataLoader

img_formats = ['bmp', 'jpg', 'jpeg', 'png', 'tif', 'tiff', 'dng', 'webp', 'mpo']  # acceptable image suffixes

def create_dataloader(args, mode):
    transform_lst = []
    transform_lst += [A.Resize(224,224)]
    if mode=='train':
        transform_lst += [A.HorizontalFlip(p=0.5)]
        transform_lst += [A.VerticalFlip(p=0.5)]
        transform_lst += [A.ShiftScaleRotate(p=0.5)]
        transform_lst += [A.Cutout(num_holes=args.num_holes, max_h_size=args.hole_size, max_w_size=args.hole_size, fill_value=114, p=0.5)]
    transform_lst += [A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225], always_apply=True)]
    transform_lst += [A.pytorch.ToTensorV2()]
    transforms = albumentations.Compose(transform_lst)

    dataset = LoadImagesAndLabels(args.data_path, mode, 224, transforms)

    dataloader = DataLoader(dataset, batch_size=args.batch_size, num_workers=0, shuffle=True)

    return dataset, dataloader

class LoadImagesAndLabels(Dataset):
    def __init__(self, data_path, mode, img_size, transform):
        self.transform = transform
        self.img_size = img_size
        self.mode = mode
        df = pd.read_csv(data_path)
        self.classes = list(set(df.Class))
        self.df = df[df['type']==mode]


    def __getitem__(self, index):
        path = self.df.iloc[index]['img_path']
        img = cv2.imread(path)
        assert img is not None, f'Image Not Found {path}'
        img = self.transform(image=img)['image']

        return img, self.classes.index(self.df.iloc[index]['Class'])

    def __len__(self):
        return len(self.df)

if __name__ == '__main__':
    args = Arguments().parser().parse_args()

    args.device = torch.device('cuda:'+str(args.gpu) if torch.cuda.is_available() else 'cpu')
    model = Classification(args)
    model.train(ckpt_path=args.ckpt_path)
    # model.test(load_ckpt=args.load_ckpt, result_path=args.result_path)
