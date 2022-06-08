import torch
import argparse
import os
import glob
from model import model_load


def model_validation(args):
    model = model_load(backbone=args.backbone, num_classes=args.num_classes, pretrained=args.pretrained)

    for i in range(args.test_num):
        rand_tensor = torch.rand(args.batch_size,3,224,224)
        output = model(rand_tensor)

        if list(output.shape)==[args.batch_size, args.num_classes]:
            print("OK Model Validate", i+1)




if __name__ ==  "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument('--backbone', type=str, default="resnet18")
    parser.add_argument('--num_classes', type=int, default=3)
    parser.add_argument('--pretrained', type=bool, default=True)
    parser.add_argument('--test_num', type=int, default=10)
    parser.add_argument('--batch_size', type=int, default=4)

    args = parser.parse_args()

    print("Model Validation...")
    model_validation(args)
