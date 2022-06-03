import pandas as pd
import argparse
import numpy as np
import os
import glob
from datetime import datetime
import xml.etree.ElementTree as ET
import warnings
from sklearn.model_selection import train_test_split
from PIL import Image

warnings.filterwarnings('ignore')

def split_dataset(input_data, output_path, img_width, img_height):
    images_path = f'{input_data}/images'
    annotations_path = f'{input_data}/annotations'
    dataset = {
                "file":[],
                "name":[],
                "width":[],
                "height":[],
                "xmin":[],
                "ymin":[],
                "xmax":[],
                "ymax":[],
               }

    for anno in glob.glob(annotations_path+"/*.xml"):
        tree = ET.parse(anno)
        for elem in tree.iter():
            if 'size' in elem.tag:
                for attr in list(elem):
                    if 'width' in attr.tag:
                        width = int(round(float(attr.text)))
                    if 'height' in attr.tag:
                        height = int(round(float(attr.text)))

            if 'object' in elem.tag:
                for attr in list(elem):

                    if 'name' in attr.tag:
                        name = attr.text
                        dataset['name']+=[name]
                        dataset['width']+=[width]
                        dataset['height']+=[height]
                        dataset['file']+=[anno.split('/')[-1][0:-4]]

                    if 'bndbox' in attr.tag:
                        for dim in list(attr):
                            if 'xmin' in dim.tag:
                                xmin = int(round(float(dim.text)))
                                dataset['xmin']+=[xmin]
                            if 'ymin' in dim.tag:
                                ymin = int(round(float(dim.text)))
                                dataset['ymin']+=[ymin]
                            if 'xmax' in dim.tag:
                                xmax = int(round(float(dim.text)))
                                dataset['xmax']+=[xmax]
                            if 'ymax' in dim.tag:
                                ymax = int(round(float(dim.text)))
                                dataset['ymax']+=[ymax]

    df=pd.DataFrame(dataset)
    df.head()

    name_dict = {
        'with_mask': 0,
        'mask_weared_incorrect': 1,
        'without_mask': 2
    }

    df['class'] = df['name'].map(name_dict)
    np.sort(df.name.unique())


    fileNames = [*os.listdir(f'{input_data}/images')]
    print('There are {} images in the dataset'.format(len(fileNames)))

    train, test = train_test_split(fileNames, test_size=0.1, random_state=22)
    test, val = train_test_split(test, test_size=0.7, random_state=22)
    print("Length of Train =",len(train))
    print("="*30)
    print("Length of Valid =",len(val))
    print("="*30)
    print("Length of test =", len(test))

    image_lst = [train, val, test]
    folder_name = ['train', 'val', 'test']

    os.makedirs(f'{output_path}', exist_ok=True)
    os.makedirs(f'{output_path}/train', exist_ok=True)
    os.makedirs(f'{output_path}/val', exist_ok=True)
    os.makedirs(f'{output_path}/test', exist_ok=True)
    os.makedirs(f'{output_path}/train/images', exist_ok=True)
    os.makedirs(f'{output_path}/train/labels', exist_ok=True)
    os.makedirs(f'{output_path}/test/images', exist_ok=True)
    os.makedirs(f'{output_path}/test/labels', exist_ok=True)
    os.makedirs(f'{output_path}/val/images', exist_ok=True)
    os.makedirs(f'{output_path}/val/labels', exist_ok=True)

    #copy data
    for i, data_type in enumerate(image_lst):
        for image in data_type:
            print(image)
            img = Image.open(f'{input_data}/images/{image}')
            img1 = img.resize((640, 480))
            _ = img1.save(f'{output_path}/{folder_name[i]}/images/{image}')

    print(df.head())

    df['xmax'] = (img_width/df['width'])*df['xmax']
    print(df['xmax'])
    df['ymax'] = (img_height/df['height'])*df['ymax']
    df['xmin'] = (img_width/df['width'])*df['xmin']
    df['ymin'] = (img_height/df['height'])*df['ymin']
    df[['xmax', 'ymax', 'xmin', 'ymin']] = df[['xmax', 'ymax', 'xmin', 'ymin']].astype('int64')
    df['x_center'] = (df['xmax']+df['xmin'])/(2*640)
    df['y_center'] = (df['ymax']+df['ymin'])/(2*480)
    df['box_height'] = (df['xmax']-df['xmin'])/(640)
    df['box_width'] = (df['ymax']-df['ymin'])/(480)
    print(df.head())
    df = df.astype('string')


    # create_labels
    for i, data_type in enumerate(image_lst):
        file_names = [x.split(".")[0] for x in data_type]

        for name in file_names:
            data = df[df.file==name]
            box_list = []

            for index in range(len(data)):
                row = data.iloc[index]
                box_list.append(row['class']+" "+row["x_center"]+" "+row["y_center"]\
                            +" "+row["box_height"]+" "+row["box_width"])

            text = "\n".join(box_list)
            with open(f'{output_path}/{folder_name[i]}/labels/{name}.txt', "w") as file:
                file.write(text)



    yaml_text = """ train: /home/jovyan/data/train/images
    val: /home/ljj/jovyan/data/val/images

    nc: 3
    names: ['with_mask', 'mask_weared_incorrect', 'without_mask']"""

    with open("/home/ljj/ws/dataset/data/data.yaml", 'w') as file:
        file.write(yaml_text)

    print("Complete Data Prepare ! ! !")



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input-path', type=str, default='/home/ljj/ws/dataset/mask', help='dataset root path')
    parser.add_argument('--output-path', type=str, default='/home/ljj/ws/dataset/data', help='dataset root path')
    parser.add_argument('--img-width', type=int, default=640, help='resize')
    parser.add_argument('--img-height', type=int, default=480, help='resize')
    opt = parser.parse_args()

    split_dataset(opt.input_path, opt.output_path, opt.img_width, opt.img_height)

