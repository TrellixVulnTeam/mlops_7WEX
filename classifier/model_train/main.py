import os
import json
import shutil
import torch
import torchvision
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms
from torchvision.utils import save_image
from torchinfo import summary
from torch.autograd import Variable
from utils import *

import numpy as np
import torchvision.utils as vutils
from dataloader import create_dataloader
from arguments import Arguments

from utils.log import *
from utils.util import *
from utils.metrics import accuracy
from models.model import model_select

from dataloader import *
import mlflow

def mlflow_run():

    return True

def run(args):
    remote_server_uri = "http://0.0.0.0:5000" # set to your server URI
    mlflow.set_tracking_uri(remote_server_uri)
    os.environ["MLFLOW_S3_ENDPOINT_URL"] = "http://0.0.0.0:9090"
    os.environ["AWS_ACCESS_KEY_ID"] = "minio"
    os.environ["AWS_SECRET_ACCESS_KEY"] = "minio123"
    mlflow.set_experiment("Animal")
    tags = {"Class": "Cats, Dogs, Pandas", "Test": "Complete"}


    with mlflow.start_run(run_name='animal', tags=tags):
        args_dict = vars(args)
        for k, v in args_dict.items():
            mlflow.log_param(k, v)
        model = model_select(backbone=args.backbone, num_classes=args.num_classes, pretrained=args.pretrained).to(device=args.device)
        model_summary = str(summary(model, input_size=(1, 3, 224, 224)))
        mlflow.log_text(model_summary, 'model_summary.txt')

        optimizer = optim.SGD(model.parameters(), lr=args.learning_rate, momentum=args.momentum, nesterov=True)

        train_datasets, train_loader = create_dataloader(args, mode='train')
        val_datasets, val_loader = create_dataloader(args, mode='valid')
        class_dict=dict(zip(range(len(train_datasets.classes)), train_datasets.classes))
        mlflow.log_dict(class_dict, 'classes.json')

        criterion = nn.CrossEntropyLoss().to(device=args.device)

        train_loss = AverageMeter()
        train_acc = AverageMeter()

        val_loss = AverageMeter()
        val_acc = AverageMeter()

        best_accuracy=0

        for epoch in range(args.epochs):
            model.train()
            for _iter, (data, label) in enumerate(train_loader):
                data = data.to(device=args.device)
                label = label.to(device=args.device)
                optimizer.zero_grad()

                output = model(data)
                loss = criterion(output, label)
                loss.backward()
                optimizer.step()

                acc_res = accuracy(output.data, label.data)
                train_loss.update(loss.item(), data.size(0))
                train_acc.update(acc_res.item(), data.size(0))

                if _iter % 50 ==49:
                    print('Epoch : {step:4d} | Train_Loss: {tr_l:.4f} | Train_Acc: {tr_A:.4f} |'.format(
                        step=epoch,
                        tr_l=train_loss.avg,
                        tr_A=train_acc.avg,
                        ))

            model.eval()
            for _iter, (data, label) in enumerate(val_loader):
                data = data.to(device=args.device)
                label = label.to(device=args.device)

                output = model(data)

                loss = criterion(output, label)
                val_loss.update(loss.item(), data.size(0))

                acc_res = accuracy(output.data, label.data)
                val_acc.update(acc_res.item(), data.size(0))

            mlflow.log_metric('Train Loss', train_loss.avg, step=epoch)
            mlflow.log_metric('Validation Loss', val_loss.avg, step=epoch)
            mlflow.log_metric('Train Accuracy', train_acc.avg, step=epoch)
            mlflow.log_metric('Validation Accuracy', val_acc.avg, step=epoch)
            print('Epoch : {step:4d} | Train_Loss: {tr_l:.4f} | Train_Acc: {tr_A:.4f} | Validation_Loss: {val_l:.4f} | Validation_Acc: {val_A:.4f}'.format(
                step=epoch,
                tr_l=train_loss.avg,
                tr_A=train_acc.avg,
                val_l=val_loss.avg,
                val_A=val_acc.avg,
                ))
            mlflow.pytorch.log_model(model, "model")


if __name__ == '__main__':
    args = Arguments().parser().parse_args()
    args.device = torch.device('cuda:'+str(args.gpu) if torch.cuda.is_available() else 'cpu')
    run(args)
