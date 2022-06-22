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

app = Flask(__name__)

dic = {0 : 'Cat', 1 : 'Dog', 2 : 'Panda'}

model_name = 'detection'
version=1
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
print(model)
# transform_lst = []
# transform_lst += [A.Resize(224,224)]
# transform_lst += [A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225], always_apply=True)]
# transform_lst += [A.pytorch.ToTensorV2()]
# transforms = albumentations.Compose(transform_lst)


# def inference(path, model):
    # img = cv2.imread(path)
    # img = transforms(image=img)['image']

    # # img = torch.from_numpy(path).float()

    # img = img.unsqueeze(0) # (H, W, C) -> (C, H, W) -> (1, C, H, W)
    # print(img.shape)

    # model = model.cuda()
    # img = img.cuda()

    # outputs = model(img)

    # out = torch.argmax(outputs, dim=1).tolist()
    # print(out)
    # return dic[out[0]]



# # routes
# @app.route("/", methods=['GET', 'POST'])
# def main():
    # return render_template("index.html")

# @app.route("/about")
# def about_page():
    # return "Please subscribe Artificial Intelligence Hub..!!!"

# @app.route("/submit", methods = ['GET', 'POST'])
# def get_output():
    # if request.method == 'POST':
        # img = request.files['my_image']
        # print(img)

        # img_path = "static/" + img.filename
        # if img_path =='static/':
            # return 404

        # print("Image Save ! ! ", img_path)
        # img.save(img_path)
        # p = inference(img_path, model)


    # return render_template("index.html", prediction = p, img_path = img_path)






# if __name__ =='__main__':
	# app.run(host='0.0.0.0', port=5001, debug=True)
