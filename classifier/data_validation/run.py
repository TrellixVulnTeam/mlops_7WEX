'''
1. Data Validation
2. Data preprocessing
'''
import argparse
import os
import glob
import base64
from PIL import Image
import io
from pathlib import Path
import pandas as pd
import cv2
from sklearn.model_selection import StratifiedShuffleSplit

def data_read(data_path):
    data_lst = glob.glob(data_path+"/**/*.png",recursive=True)
    print(len(data_lst))
    return data_lst


def data_validation(data_lst):
    data = ['train', 'valid']
    for data in data_lst:
        # If Input is binary file
        # try:
        #     image = base64.b64decode(data)
        #     img = Image.open(io.BytesIO(image))
        # except:
        #     raise Exception('file is not valid base64 image')
        p = Path(data)

        if p.suffix[1:] in ['bmp', 'jpg', 'jpeg', 'png', 'tif', 'tiff', 'dng', 'webp', 'mpo']:
            print(data)
            img = cv2.imread(data)
            width, height, channel = img.shape
            # if width < 2000 and height < 2000 and channel == 3:
                # print("Validate Data .. {path:%s}".format(path=p))
            # else:
                # raise Exception('Image size exceeded, width and height must be less than 2000 pixels and channel is 3')
        else:
            raise Exception("Image is not valid, Only 'base64' and image (bmp, ... ) format is valid")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument('--data_path', type=str, default="/home/jongjin/st/dataset/images/train/images")
    parser.add_argument('--image_width', type=int, default=224)
    parser.add_argument('--image_height', type=int, default=224)
    parser.add_argument('--image_channel', type=int, default=3)

    args = parser.parse_args()

    print("Data Read...")
    data_lst=data_read(args.data_path)

    print("Data Validation...")
    data_validation(data_lst)
