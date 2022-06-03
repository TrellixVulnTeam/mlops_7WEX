'''
- Image Validation
1. Cheching base64 image string validation_data
2. Checking format of image that I want to support
3. Checking image dimentions
- Data Processing, Availalbe data format ( In pandas )
1. Make data frame list
2. Make pandas dataframe
3. Data Split
4. Save Data frame
'''

import os
import splitfolders
import glob
import base64
from PIL import Image
import io
from pathlib import Path
import pandas as pd

def data_read(data_path):
    data_lst = glob.glob(data_path+"**/*.jpg",recursive=True)
    print(len(data_lst))


def data_validation(data_lst):
    data = ['train', 'valid']
    for data in data_lst:
        try:
            image = base64.b64decode(data)
            img = Image.open(io.BytesIO(image))
        except:
            raise Exception('file is not valid base64 image')

        if img.format.lower() in ['bmp', 'jpg', 'jpeg', 'png', 'tif', 'tiff', 'dng', 'webp', 'mpo']:
            width, height = img.size
            print(img.size)
            if width < 2000 and height < 2000:
                return True
            else:
                raise Exception('Image size exceeded, width and height must be less than 2000 pixels and channel is 3')
        else:
            raise Exception("Image is not valid, Only 'base64' and image (bmp, ... ) format is valid")

def data_processing(data_lst):
    df_lst = []
    for data in data_lst:
        classes = data.split('/')[-2]
        df_lst.append([str(data), classes])
    df = pd.DataFrame(df_lst, columns=["img_path", "class"])

    df.split



        df_lst.append()





    # label = os.listdir(data_path)
    # splitfolders.ratio(f'{data_path}', output=f'{data_path}/dataset', seed=42, ratio=(0.7, 0.15, 0.15))
    # # print data amount
    # for i in data:
        # for j in label:
            # count = len(os.listdir(f'{data_path}/dataset/{i}/{j}'))
            # print(f'Crack | {i} | {j} : {count}')

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument('--data_path', type=str, default="/Users/kaejong/workspace/datasets/animals")
    parser.add_argument('--image_width', type=int, default=224)
    parser.add_argument('--image_height', type=int, default=224)
    parser.add_argument('--image_channel', type=int, default=3)
    parser.add_argument('--npy_interval', type=int, default=500)

    args = parser.parse_args()

    print("Data Read...")
    data_read(args.data_path)


    print("Image Resize...")
    image_resize(data_path=opt.data_path)
    print("Data Split...")
    data_split(data_path=opt.data_path)
    print("Validating data...")
    validation_data(
        train_data_file=os.path.join(args.train_data_path, args.train_data_file),
        test_data_file=os.path.join(args.test_data_path, args.test_data_file),
        faiss_train_data_file=os.path.join(args.faiss_train_data_path, args.faiss_train_data_file),
        faiss_test_data_file=os.path.join(args.faiss_test_data_path, args.faiss_test_data_file),
        image_width=args.image_width,
        image_height=args.image_height,
        image_channel=args.image_channel,
        image_type=np.float32,
        label_type=np.int64
    )
