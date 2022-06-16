import os
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
    mlflow.set_tracking_uri("http://210.123.42.43:5000")
    client = mlflow.tracking.MlflowClient("http://210.123.42.43:5000")

    exp = mlflow.get_experiment_by_name("Classification")
    if exp is None:
        exp_id = mlflow.create_experiment("Classification")
    else:
        exp_id = exp.experiment_id

    mlflow_run = mlflow.start_run(run_name = 'Animals', experiment_id=exp_id)

    return True

def run(args):
    model = model_select(backbone=args.backbone, num_classes=args.num_classes, pretrained=args.pretrained).to(device=args.device)
    model_summary = summary(model, input_size=(1, 3, 224, 224))
    optimizer = optim.SGD(model.parameters(), lr=args.learning_rate, momentum=args.momentum, nesterov=True)


    train_datasets, train_loader = create_dataloader(args, mode='train')
    val_datasets, val_loader = create_dataloader(args, mode='valid')

    criterion = nn.CrossEntropyLoss().to(device=args.device)

    train_loss = AverageMeter()
    train_acc = AverageMeter()

    val_loss = AverageMeter()
    val_acc = AverageMeter()

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

            if _iter % 50 ==49:
                print('Epoch : {step:4d} | valid_Loss: {tr_l:.4f} | Valid_Accuracy: {tr_A:.4f} |'.format(
                    step=epoch,
                    tr_l=val_loss.avg,
                    tr_A=val_acc.avg,
                    ))

        print('Epoch : {step:4d} | Train_Loss: {tr_l:.4f} | Train_Acc: {tr_A:.4f} | Validation_Loss: {val_l:.4f} | Validation_Acc: {val_A:.4f}'.format(
            step=epoch,
            tr_l=train_loss.avg,
            tr_A=train_acc.avg,
            val_l=val_loss.avg,
            val_A=val_acc.avg,
            ))

        if args.mlflow_status:
            mlflow.log_metric('Train Loss', train_loss.avg)
            mlflow.log_metric('Validation Loss', val_loss.avg)
            mlflow.log_metric('Train Accuracy', train_acc.avg)
            mlflow.log_metric('Validation Accuracy', val_acc.avg)


if __name__ == '__main__':
    args = Arguments().parser().parse_args()
    args.device = torch.device('cuda:'+str(args.gpu) if torch.cuda.is_available() else 'cpu')
    args.mlflow_status = True
    args.mlflow_status = mlflow_run()
    args_dict = vars(args)
    for k, v in args_dict.items():
        if args.mlflow_status:
            if k == 'mlflow_status':
                pass
            else:
                mlflow.log_param(k, v)
        else:
            print(k, v)
    run(args)
    # model.test(load_ckpt=args.load_ckpt, result_path=args.result_path)
