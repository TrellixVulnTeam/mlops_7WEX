import torch
import os
import mlflow
from mlflow.tracking import MlflowClient
import torch.nn.functional as F
import cv2

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

# def inference(img):
    # imgs = torch.tensor(img).permute(0,3,1,2)
    # imgs = F.interpolate(imgs, size=224) / 255.0
    # outputs = artifacts.pytorch.model(imgs)

    # return torch.argmax(outputs, dim=1).tolist()

path = '/home/jongjin/st/dataset/animals_resize/cats_00001.jpg'
img = cv2.imread(path)
print(img.shape)

model = load_model(model_name='animal', version=1)

print(model)

