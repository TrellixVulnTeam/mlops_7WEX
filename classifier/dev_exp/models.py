from torchvision import datasets, models, transforms
import torch.nn as nn
import torch
import os 
from pathlib import Path
def ResNet(backbone, num_classes, weights_path):
    if backbone == 'RESNET18':
        model = models.resnet18(pretrained=False)
    elif backbone == 'RESNET34':
        model = models.resnet34(pretrained=False)
    elif backbone == 'RESNET50':
        model = models.resnet50(pretrained=False)

    model.load_state_dict(torch.load(weights_path), strict=False)

    num_ftrs = model.fc.in_features
    block =[]
    block += [nn.Linear(num_ftrs, 1024)]
    block += [nn.Linear(1024, 512)]
    block += [nn.Linear(512, num_classes)]
    model.fc = nn.Sequential(*block)
    
    return model

'''
model_urls = {
    'resnet18': 'https://download.pytorch.org/models/resnet18-5c106cde.pth',
    'resnet34': 'https://download.pytorch.org/models/resnet34-333f7ec4.pth',
    'resnet50': 'https://download.pytorch.org/models/resnet50-19c8e357.pth',
    'resnet101': 'https://download.pytorch.org/models/resnet101-5d3b4d8f.pth',
    'resnet152': 'https://download.pytorch.org/models/resnet152-b121ed2d.pth',
}
'''