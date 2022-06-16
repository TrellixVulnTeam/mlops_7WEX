from torchvision import datasets, models, transforms
import os
import torch.nn as nn
import torch

import torch
import torch.nn as nn
from torchvision import models
from pathlib import Path

######################################################################################################################
#                                             VGG Begin
######################################################################################################################
'''
# ConvNet Configuration
                A | A-LRN | B | C | D | E
weight layer    11  11      13  16  16  19
'''

VGG_ConvNet_Configuration={
    'VGG11':[64, 'P', 128, 'P', 256, 256, 'P', 512, 512, 'P', 512, 512, 'P'],
    'VGG13':[64, 64, 'P', 128, 128, 'P', 256, 256, 'P', 512, 512, 'P', 512, 512, 'P'],
    'VGG16':[64, 64, 'P', 128, 128, 'P', 256, 256, '256', 'P', 512, 512, '512', 'P', 512, 512, '512', 'P'],
    'VGG19':[64, 64, 'P', 128, 128, 'P', 256, 256, 256, 256, 'P', 512, 512, 512, 512, 'P', 512, 512, 512, 512, 'P'],
}

class VGG(nn.Module):
    def __init__(self, in_channels, num_classes):
        super(VGG, self).__init__()
        self.in_channels = in_channels
        self.vgg_version = 'VGG19'
        self.backbone = self.create_VGG(VGG_ConvNet_Configuration[self.vgg_version])

        self.classifier = nn.Sequential(
            nn.Linear(512*7*7, 4096),
            nn.ReLU(),
            nn.Dropout(),
            nn.Linear(4096, 1024),
            nn.ReLU(),
            nn.Dropout(),
            nn.Linear(1024, num_classes),
        )
        self._initialize_weights()

    def forward(self, x):
        x = self.backbone(x)
        x = x.view(x.size(0), -1)
        out = self.classifier(x)
        return out


    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)


    def create_VGG(self, architecture):
        layer = []
        in_channels = self.in_channels
        for l in architecture:
            if l == 'P':
                layer += [nn.MaxPool2d(2)]
            else:
                if l == '256' or l == '512':
                    l = int(l)
                    layer += [nn.Conv2d(in_channels, l, kernel_size=1, stride=1, padding=0), nn.BatchNorm2d(l), nn.ReLU()]
                else:
                    layer += [nn.Conv2d(in_channels, l, kernel_size=3, stride=1, padding=1), nn.BatchNorm2d(l), nn.ReLU()]
                in_channels = l

        return nn.Sequential(*layer)
######################################################################################################################
#                                             VGG End
######################################################################################################################
