from flask import Flask, render_template, request
import torch
import os
import mlflow
from mlflow.tracking import MlflowClient
import cv2
import albumentations as A
import albumentations.pytorch
# from keras.models import load_model
# from keras.preprocessing import image
from torch.autograd import Variable
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from yolo.utils.utils import *



app = Flask(__name__)

model_name = 'detection'
version=2
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
model = mlflow.pytorch.load_model(model_uri)

classes = load_classes("/home/jongjin/st/mlops/service/flask_detection/yolo/df2cfg/df2.names")

cmap = plt.get_cmap("rainbow")
colors = np.array([cmap(i) for i in np.linspace(0, 1, 13)])



path = '/home/jongjin/st/asdf.jpeg'
def inference(path, model):
    img = cv2.imread(path)
    dets = get_detections(img, model)

    if len(dets) != 0 :
            dets.sort(reverse=False ,key = lambda x:x[4])
            for x1, y1, x2, y2, cls_conf, cls_pred in dets:
                    print("\t+ Label: %s, Conf: %.5f" % (classes[int(cls_pred)], cls_conf))

                    color = colors[int(cls_pred)]

                    color = tuple(c*255 for c in color)
                    color = (.7*color[2],.7*color[1],.7*color[0])

                    font = cv2.FONT_HERSHEY_SIMPLEX


                    x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                    text =  "%s conf: %.3f" % (classes[int(cls_pred)] ,cls_conf)

                    cv2.rectangle(img,(x1,y1) , (x2,y2) , color,3)
                    y1 = 0 if y1<0 else y1
                    y1_rect = y1-25
                    y1_text = y1-5

                    if y1_rect<0:
                        y1_rect = y1+27
                        y1_text = y1+20
                    cv2.rectangle(img,(x1-2,y1_rect) , (x1 + int(8.5*len(text)),y1) , color,-1)
                    cv2.putText(img,text,(x1,y1_text), font, 0.5,(255,255,255),1,cv2.LINE_AA)
            cv2.imwrite('/home/jongjin/st/mlops/service/flask_detection/static/ouput-test_{}_{}.jpg'.format('yolov3', '1'),img)
inference(path, model)

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
        print(img)

        img_path = "static/" + img.filename
        if img_path =='static/':
            return 404

        print("Image Save ! ! ", img_path)
        img.save(img_path)
        p = inference(img_path, model)


    return render_template("index.html", prediction = p, img_path = img_path)