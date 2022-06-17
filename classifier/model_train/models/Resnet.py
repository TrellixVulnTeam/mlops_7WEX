from lib2to3.pgen2.pgen import NFAState
import torch
import torch.nn as nn
import torch.nn.functional as F
#####################################################################################################################
                                            # Residual Network Begin
#####################################################################################################################

class ConvModule(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=True, norm='batchnorm', act='relu'):
        super().__init__()
        layers = []
        layers += [nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride, padding=padding, bias=bias)]
        layers += [nn.BatchNorm2d(num_features=out_channels) if norm == 'batchnorm' else nn.InstanceNorm2d(num_features=out_channels)]
        layers += [nn.ReLU() if act == 'relu' else nn.LeakyReLU(0.2)]

        self.block = nn.Sequential(*layers)

    def forward(self, x):
        return self.block(x)

class ResBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=True, norm='batchnorm', act='relu'):
        super().__init__()
        expansion=1

        in_channels = int(in_channels/2) if stride==2 else in_channels
        layers = []
        layers += [ConvModule(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride, padding=padding, bias=bias, norm=norm, act=act)]
        layers += [ConvModule(in_channels=out_channels, out_channels=out_channels, kernel_size=kernel_size, stride=1, padding=padding, bias=bias, norm=norm, act=None)]
        self.resblock = nn.Sequential(*layers)

        # identity mapping, input과 output의 feature map size, filter 수가 동일한 경우 사용.
        self.shortcut = nn.Sequential()
        self.relu = nn.ReLU()

        # projection mapping using 1x1conv
        if stride != 1 or in_channels != expansion * out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels * expansion, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels * expansion)
            )

    def forward(self, x):
        return x + self.resblock(x)

    def forward(self, x):
        x = self.resblock(x) + self.shortcut(x)
        x = self.relu(x)
        return x


class ResNet(nn.Module):
    def __init__(self, in_channels, num_classes, block=[2,2,2,2]):
        super(ResNet, self).__init__()
        block_feat = [64, 128, 256, 512]

        self.layer1 = ConvModule(in_channels, out_channels=block_feat[0], kernel_size=7, stride=2, padding=3, bias=False, norm='batchnorm', act='relu')
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        residual = []
        for i, n in enumerate(block):
            for j in range(n):
                stride = 2 if (j==0 and i!=0) else 1
                residual += [ResBlock(block_feat[i], block_feat[i], kernel_size=3, stride=stride, padding=1, bias=True, norm='batchnorm', act='relu')]
            
        self.res = nn.Sequential(*residual)

        # self.layer2 = ConvModule(block_feat[-1], block_feat[-1], kernel_size=3, stride=1, padding=1, bias=True, norm='batchnorm', act='relu')
        self.avgpool = nn.AdaptiveAvgPool2d((1,1))

        self.classifier = nn.Sequential(
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(),
            nn.Linear(256, num_classes),
        )

        # self._initialize_weights()


    def forward(self, x):
        x = self.layer1(x)
        x = self.maxpool(x)
        x = self.res(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)

        return x

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





# #####################################################################################################################
#                                             # Residual Network Begin
# #####################################################################################################################
# class BasicBlock(nn.Module):
#     expansion = 1
#     def __init__(self, in_channels, out_channels, stride=1):
#         super().__init__()

#         # BatchNorm에 bias가 포함되어 있으므로, conv2d는 bias=False로 설정합니다.
#         self.residual_function = nn.Sequential(
#             nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False),
#             nn.BatchNorm2d(out_channels),
#             nn.ReLU(),
#             nn.Conv2d(out_channels, out_channels * BasicBlock.expansion, kernel_size=3, stride=1, padding=1, bias=False),
#             nn.BatchNorm2d(out_channels * BasicBlock.expansion),
#         )

#         # identity mapping, input과 output의 feature map size, filter 수가 동일한 경우 사용.
#         self.shortcut = nn.Sequential()

#         self.relu = nn.ReLU()

#         # projection mapping using 1x1conv
#         if stride != 1 or in_channels != BasicBlock.expansion * out_channels:
#             self.shortcut = nn.Sequential(
#                 nn.Conv2d(in_channels, out_channels * BasicBlock.expansion, kernel_size=1, stride=stride, bias=False),
#                 nn.BatchNorm2d(out_channels * BasicBlock.expansion)
#             )

#     def forward(self, x):
#         x = self.residual_function(x) + self.shortcut(x)
#         x = self.relu(x)
#         return x


# class BottleNeck(nn.Module):
#     expansion = 4
#     def __init__(self, in_channels, out_channels, stride=1):
#         super().__init__()

#         self.residual_function = nn.Sequential(
#             nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, bias=False),
#             nn.BatchNorm2d(out_channels),
#             nn.ReLU(),
#             nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False),
#             nn.BatchNorm2d(out_channels),
#             nn.ReLU(),
#             nn.Conv2d(out_channels, out_channels * BottleNeck.expansion, kernel_size=1, stride=1, bias=False),
#             nn.BatchNorm2d(out_channels * BottleNeck.expansion),
#         )

#         self.shortcut = nn.Sequential()

#         self.relu = nn.ReLU()

#         if stride != 1 or in_channels != out_channels * BottleNeck.expansion:
#             self.shortcut = nn.Sequential(
#                 nn.Conv2d(in_channels, out_channels*BottleNeck.expansion, kernel_size=1, stride=stride, bias=False),
#                 nn.BatchNorm2d(out_channels*BottleNeck.expansion)
#             )

#     def forward(self, x):
#         x = self.residual_function(x) + self.shortcut(x)
#         x = self.relu(x)
#         return x

# class ResNet(nn.Module):
#     def __init__(self, block, num_block, num_classes=10, init_weights=True):
#         super().__init__()

#         self.in_channels=64

#         self.conv1 = nn.Sequential(
#             nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False),
#             nn.BatchNorm2d(64),
#             nn.ReLU(),
#             nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
#         )

#         self.conv2_x = self._make_layer(block, 64, num_block[0], 1)
#         self.conv3_x = self._make_layer(block, 128, num_block[1], 2)
#         self.conv4_x = self._make_layer(block, 256, num_block[2], 2)
#         self.conv5_x = self._make_layer(block, 512, num_block[3], 2)

#         self.avg_pool = nn.AdaptiveAvgPool2d((1,1))
#         self.fc = nn.Linear(512 * block.expansion, num_classes)

#         # weights inittialization
#         if init_weights:
#             self._initialize_weights()

#     def _make_layer(self, block, out_channels, num_blocks, stride):
#         strides = [stride] + [1] * (num_blocks - 1)
#         layers = []
#         for stride in strides:
#             layers.append(block(self.in_channels, out_channels, stride))
#             self.in_channels = out_channels * block.expansion

#         return nn.Sequential(*layers)

#     def forward(self,x):
#         output = self.conv1(x)
#         output = self.conv2_x(output)
#         x = self.conv3_x(output)
#         x = self.conv4_x(x)
#         x = self.conv5_x(x)
#         x = self.avg_pool(x)
#         x = x.view(x.size(0), -1)
#         x = self.fc(x)
#         return x

#     # define weight initialization function
#     def _initialize_weights(self):
#         for m in self.modules():
#             if isinstance(m, nn.Conv2d):
#                 nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
#                 if m.bias is not None:
#                     nn.init.constant_(m.bias, 0)
#             elif isinstance(m, nn.BatchNorm2d):
#                 nn.init.constant_(m.weight, 1)
#                 nn.init.constant_(m.bias, 0)
#             elif isinstance(m, nn.Linear):
#                 nn.init.normal_(m.weight, 0, 0.01)
#                 nn.init.constant_(m.bias, 0)

# ######################################################################################################################
# #                                             Residual Network End
# ######################################################################################################################
