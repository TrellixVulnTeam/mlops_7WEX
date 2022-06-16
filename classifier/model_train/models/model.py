from torchvision import datasets, models, transforms
import os
import torch.nn as nn
import torch
import torch
import torch.nn as nn
from torchvision import models
from pathlib import Path
from models.VGG import *
from models.Resnet import *

def model_select(backbone, num_classes, pretrained=False):
    if pretrained==True:
        if backbone == 'resnet18':
            model = models.resnet18(pretrained=False)
        elif backbone == 'resnet34':
            model = models.resnet34(pretrained=False)
        elif backbone == 'resnet50':
            model = models.resnet50(pretrained=False)

        model_path = Path(os.path.abspath(__file__)).parent / (backbone+'.pth')
        model.load_state_dict(torch.load(model_path), strict=False)

        num_ftrs = model.fc.in_features
        block =[]
        block += [nn.Linear(num_ftrs, 1024)]
        block += [nn.Linear(1024, 512)]
        block += [nn.Linear(512, num_classes)]
        model.fc = nn.Sequential(*block)

    else:
        if backbone == 'resnet18':
            # model=ResNet(BasicBlock, [2,2,2,2], num_classes)
            model=ResNet(3, num_classes, [2,2,2,2])
        # elif backbone == 'resnet34':
        #     model = ResNet(BasicBlock, [3, 4, 6, 3], num_classes)
        # elif backbone == 'resnet50':
        #     model = ResNet(BottleNeck, [3,4,6,3], num_classes)
        elif backbone == 'vgg':
            model = VGG(3, num_classes)

        
    return model
