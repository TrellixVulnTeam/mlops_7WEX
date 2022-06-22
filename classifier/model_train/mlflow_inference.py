import torch
import os
import mlflow
from mlflow.tracking import MlflowClient
import torch.nn.functional as F
import cv2
import json
import albumentations as A
import albumentations.pytorch

transform_lst = []
transform_lst += [A.Resize(224,224)]
# if mode=='train':
    # transform_lst += [A.HorizontalFlip(p=0.5)]
    # transform_lst += [A.VerticalFlip(p=0.5)]
    # transform_lst += [A.ShiftScaleRotate(p=0.5)]
    # transform_lst += [A.Cutout(num_holes=args.num_holes, max_h_size=args.hole_size, max_w_size=args.hole_size, fill_value=114, p=0.5)]
transform_lst += [A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225], always_apply=True)]
transform_lst += [A.pytorch.ToTensorV2()]
transforms = albumentations.Compose(transform_lst)
def load_model(model_name, version):
    print(model_name, version)
    os.environ["MLFLOW_S3_ENDPOINT_URL"] = "http://0.0.0.0:9090"
    os.environ["AWS_ACCESS_KEY_ID"] = "minio"
    os.environ["AWS_SECRET_ACCESS_KEY"] = "minio123"
    client = MlflowClient("http://0.0.0.0:5000")
    print(client)

    filter_string = "name='{}'".format(model_name)

    results=client.search_model_versions(filter_string)
    print(results)

    for res in results:
        if res.version == str(version):
            model_uri = res.source
            break

    print(model_uri)
    reconstructed_model = mlflow.pytorch.load_model(model_uri)

    return reconstructed_model

def inference(path, model):
    img = cv2.imread(path)
    img = transforms(image=img)['image']

    # img = torch.from_numpy(path).float()

    img = img.unsqueeze(0) # (H, W, C) -> (C, H, W) -> (1, C, H, W)
    print(img.shape)

    model = model.cuda()
    img = img.cuda()

    outputs = model(img)

    return torch.argmax(outputs, dim=1).tolist()

path = '/home/jongjin/st/dataset/animals_resize/panda_00002.jpg'

model = load_model(model_name='Animals', version=5)
res = inference(path, model)

lst = {0:'cat',1: 'dog',2: 'panda'}
print(res[0])

print(lst[res[0]])


if __name__ == '__main__':
    args = Arguments().parser().parse_args()
    args.device = torch.device('cuda:'+str(args.gpu) if torch.cuda.is_available() else 'cpu')
    run(args)
