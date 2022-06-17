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
    data_lst = glob.glob(data_path+"/**/*.jpg",recursive=True)
    print(len(data_lst))
    return data_lst


def image_resize(img):
    output_size = 224
    (h, w) = img.shape[:2]
    shape = img.shape[:2]  # current shape [height, width]
    r = output_size / max(h, w)
    if r != 1:
        img = cv2.resize(img, (int(w*r), int(h*r)), interpolation=cv2.INTER_LINEAR)

    # Scale ratio (new / old)
    r = min(output_size / h, output_size / w)

    ratio = r, r  # width, height ratios
    new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
    dw, dh = output_size - new_unpad[0], output_size - new_unpad[1]  # wh padding

    dw /= 2  # divide padding into 2 sides
    dh /= 2

    if shape[::-1] != new_unpad:  # resize
        img = cv2.resize(img, new_unpad, interpolation=cv2.INTER_LINEAR)
    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    img = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=(114,114,114))  # add border

    return img


def data_split(df, train_ratio, valid_ratio, test_ratio):
    test_size = (valid_ratio + test_ratio)

    split = StratifiedShuffleSplit(n_splits=1, test_size=test_size, random_state=1004)
    for train_idx, test_idx in split.split(df, df.Class):
        trainset = df.iloc[train_idx]
        testset  = df.iloc[test_idx]
    trainset = trainset.copy()
    testset  = testset.copy()
    trainset.loc[:,'type'] = 'train'
    test_size_ = test_ratio / (valid_ratio+test_ratio)

    split_ = StratifiedShuffleSplit(n_splits=1, test_size=test_size_, random_state=1004)
    for valid_idx, testcase_idx in split_.split(testset, testset.Class):
        validset = testset.iloc[valid_idx]
        testcaseset = testset.iloc[testcase_idx]
    validset = validset.copy()
    testcaseset = testcaseset.copy()

    validset.loc[:,'type'] = 'valid'
    testcaseset.loc[:,'type'] = 'test'
    return pd.concat([trainset, validset, testcaseset], ignore_index=True)


def data_processing(data_lst, args):
    df_lst = []
    os.makedirs(args.output_dir, exist_ok=True)
    for i, data in enumerate(data_lst):
        try:
            print('Processing .. Number : {num:d} | Data Path : {data:s}'.format(num=i+1, data=data))
            img = cv2.imread(data)
            img = image_resize(img)

            classes = data.split('/')[-2]
            resized_img_path = os.path.join(args.output_dir, data.split('/')[-1])

            cv2.imwrite(resized_img_path, img, [cv2.IMWRITE_JPEG_QUALITY, 100])
            df_lst.append([resized_img_path, classes])

        except:
            print('No Processing .. Check Data ! Path : {data:s}'.format(num=i+1, data=data))

    df = pd.DataFrame(df_lst, columns=["img_path", "Class"])
    df = data_split(df, train_ratio=args.train_ratio, valid_ratio=args.valid_ratio, test_ratio=args.test_ratio)

    df.to_csv(os.path.join(args.output_dir, (args.data_name+'.csv')))

    return True



if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument('--data_path', type=str, default="/home/jongjin/workspace/datasets/animals")
    parser.add_argument('--image_width', type=int, default=224)
    parser.add_argument('--image_height', type=int, default=224)
    parser.add_argument('--output_dir', type=str, default="/home/jongjin/workspace/datasets/animals_resize")
    parser.add_argument('--train_ratio', type=float, default=0.8)
    parser.add_argument('--valid_ratio', type=float, default=0.1)
    parser.add_argument('--test_ratio', type=float, default=0.1)
    parser.add_argument('--data_name', type=str, default='animals')

    args = parser.parse_args()

    print("Data Read...")
    data_lst=data_read(args.data_path)

    print("Data Processing...")
    data_processing(data_lst, args)
