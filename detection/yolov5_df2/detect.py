import argparse
import os
import sys
from pathlib import Path
import torch
from utils.dataloaders import IMG_FORMATS, LoadImages
from utils.general import (LOGGER, check_file, check_img_size, colorstr, cv2, increment_path, non_max_suppression, print_args, scale_coords, xyxy2xywh)
from utils.plots import Annotator, colors, save_one_box
from mlflow.tracking import MlflowClient
import mlflow


@torch.no_grad()
def run():
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
    source = '/home/jongjin/st/asdf.jpeg'
    device = 'cuda:0'
    imgsz=(416, 416)
    line_thickness=2  # bounding box thickness (pixels)
    hide_labels=False
    hide_conf=False
    model_mlflow.eval()
    stride, names = int(model_mlflow.stride[-1]), model_mlflow.names
    imgsz = check_img_size(imgsz, s=stride)  # check image size

    dataset = LoadImages(source, img_size=imgsz, stride=stride, auto=True)

    #--- Detections ---#
    for path, img, im0s, vid_cap, s in dataset:
        img = torch.from_numpy(img).to(device)
        img = img.half()
        img /= 255
        if len(img.shape) == 3:
            img = img[None]  # expand for batch dim

        pred_mlflow, _ = model_mlflow(img)
        pred_mlflow = non_max_suppression(pred_mlflow, conf_thres=0.25, iou_thres=0.4, classes=None, agnostic=False, max_det=1000)

        for i, det in enumerate(pred_mlflow):  # per image
            p, im0, frame = path, im0s.copy(), getattr(dataset, 'frame', 0)

            p = Path(p)  # to Path
            save_path = os.path.join(save_dir, str(p.name))  # im.jpg
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
    #--- Detections ---#

    LOGGER.info(f"Results saved to {colorstr('bold', save_dir)}{s}")



if __name__ == "__main__":
    run()
