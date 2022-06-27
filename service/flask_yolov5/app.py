from flask import Flask, render_template, request

import mlflow
from mlflow.tracking import MlflowClient

import os
import sys
from pathlib import Path

import cv2
import numpy as np
import matplotlib.pyplot as plt

from torch.autograd import Variable
import torch.nn.functional as F
import torch

from utils.dataloaders import LoadImages
from utils.general import (check_img_size, cv2, non_max_suppression, scale_coords)
from utils.plots import Annotator, colors

app = Flask(__name__)

#--- MLflow ---#
model_name = 'detection'
version=6
os.environ["MLFLOW_S3_ENDPOINT_URL"] = "http://0.0.0.0:9090"
os.environ["AWS_ACCESS_KEY_ID"] = "minio"
os.environ["AWS_SECRET_ACCESS_KEY"] = "minio123"
client = MlflowClient("http://0.0.0.0:5000")
filter_string = "name='{}'".format(model_name)
results=client.search_model_versions(filter_string)
for res in results:
    if res.version == str(version):
        model_uri = res.source
        break
model_mlflow = mlflow.pytorch.load_model(model_uri).cuda()
#--- MLflow ---#

save_dir = 'static/'
device = 'cuda:0'
imgsz=(600, 600)
line_thickness=1  # bounding box thickness (pixels)
hide_labels=False
hide_conf=True
model_mlflow.eval()
stride = int(model_mlflow.stride[-1])
names = ['short sleeve top', 'long sleeve top', 'short sleeve outwear', 'long sleeve outwear', 'vest', 'sling', 'shorts', 'trousers', 'skirt', 'short sleeve dress', 'long sleeve dress', 'vest dress', 'sling dress']
imgsz = check_img_size(imgsz, s=stride)  # check image size


def inference(path, model):
#--- Detections ---#
    dataset = LoadImages(path, img_size=imgsz, stride=stride, auto=True)
    for path, img, im0s, vid_cap, s in dataset:
        img = torch.from_numpy(img).to(device)
        img = img.half()
        img /= 255
        if len(img.shape) == 3:
            img = img[None]  # expand for batch dim

        pred_mlflow, _ = model_mlflow(img)
        pred_mlflow = non_max_suppression(pred_mlflow, conf_thres=0.23, iou_thres=0.4, classes=None, agnostic=False, max_det=1000)

        for i, det in enumerate(pred_mlflow):  # per image
            p, im0, frame = path, im0s.copy(), getattr(dataset, 'frame', 0)

            p = Path(p)  # to Path
            save_path = 'static/output_{}'.format(p.name)
            s += '%gx%g ' % img.shape[2:]  # print string
            gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]  # normalization gain whwh
            annotator = Annotator(im0, line_width=line_thickness, example=str(names))
            if len(det):
                # Rescale boxes from img_size to im0 size
                det[:, :4] = scale_coords(img.shape[2:], det[:, :4], im0.shape).round()

                # Print results
                for c in det[:, -1].unique():
                    n = (det[:, -1] == c).sum()  # detections per class
                    s += f"{n} {names[int(c)]}{'s' * (n > 1)}, "  # add to string

                for *xyxy, conf, cls in reversed(det):

                    c = int(cls)
                    label = None if hide_labels else (names[c] if hide_conf else f'{names[c]} {conf:.2f}')
                    annotator.box_label(xyxy, label, color=colors(c, True))

            im0 = annotator.result()
            cv2.imwrite(save_path, im0)
    if os.path.isfile(save_path):
        return 1, save_path
    else:
        return 0, None
#--- Detections ---#

# routes
@app.route("/", methods=['GET', 'POST'])
def main():
    return render_template("index.html")

@app.route("/about")
def about_page():
    return "Please subscribe Artificial Intelligence Hub..!!!"

@app.route("/submit", methods = ['GET', 'POST'])
def get_output():
    if request.method == 'POST':
        img = request.files['my_image']

        img_path = "static/" + img.filename
        if img ==None:
            return 405

        print("Image Save ! ! ", img_path)
        img.save(img_path)
        ok, save_path = inference(img_path, model_mlflow)
        print(ok, save_path)

    return render_template("index.html", prediction = ok, img_path = img_path, dst_path = save_path)



if __name__ =='__main__':
    app.run(host='0.0.0.0', port=5003, debug=True)
