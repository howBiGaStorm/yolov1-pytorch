# -*- coding: utf-8 -*-
import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable


class DarkNet(nn.Module):
    def __init__(self, pretrain=False):
        super(DarkNet, self).__init__()

        self.pretrain = pretrain

        self.net = nn.Sequential(
            # nn.Upsample(scale_factor=2),
            nn.Conv2d(3, 64, 7, stride=2, padding=3),  # 通道数3_>64,7*7卷积核，步长2，padding = 3
            nn.LeakyReLU(0.1, inplace=True),
            nn.MaxPool2d(2),  # 2*2卷积核，默认步长2

            nn.Conv2d(64, 192, 3, padding=1),
            nn.LeakyReLU(0.1, inplace=True),
            nn.MaxPool2d(2),

            nn.Conv2d(192, 128, 1),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(128, 256, 3, padding=1),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(256, 256, 1),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(256, 512, 3, padding=1),
            nn.LeakyReLU(0.1, inplace=True),
            nn.MaxPool2d(2),

            nn.Conv2d(512, 256, 1),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(256, 512, 3, padding=1),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(512, 256, 1),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(256, 512, 3, padding=1),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(512, 256, 1),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(256, 512, 3, padding=1),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(512, 256, 1),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(256, 512, 3, padding=1),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(512, 512, 1),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(512, 1024, 3, padding=1),
            nn.LeakyReLU(0.1, inplace=True),
            nn.MaxPool2d(2),

            nn.Conv2d(1024, 512, 1),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(512, 1024, 3, padding=1),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(1024, 512, 1),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(512, 1024, 3, padding=1),
            nn.LeakyReLU(0.1, inplace=True)
        )
        if self.pretrain:
            self.fc = nn.Linear(1024, 1000)

    def forward(self, x):
        output = self.net(x)
        if self.pretrain:
            output = F.avg_pool2d(output, (output.size(2), output.size(3)))
            output = output.squeeze()
            output = F.softmax(self.fc(output))
        return output


class YOLO(nn.Module):
    def __init__(self, model=None, input_size=(448, 448)):
        super(YOLO, self).__init__()
        C = 20

        ch = 512
        if model is None:  # 如果没有预训练模型，就用darknet
            ch = 1024
            model = DarkNet()

        self.features = model

        self.yolo = nn.Sequential(
            nn.Conv2d(ch, 1024, 3, padding=1),
            nn.LeakyReLU(0.1),
            nn.Conv2d(1024, 1024, 3, stride=2, padding=1),
            nn.LeakyReLU(0.1),

            nn.Conv2d(1024, 1024, 3, padding=1),
            nn.LeakyReLU(0.1),
            nn.Conv2d(1024, 1024, 3, padding=1),
            nn.LeakyReLU(0.1)
            )

        self.flatten = Flatten()  # 把结果展开，因为要对接全连接层的格式
        # self.fc1 = nn.Linear(math.ceil(input_size[0]/32) * math.ceil(input_size[1] / 32) *1024, 4096)  # math.ceil 返回数字的上入整数
        self.fc1 = nn.Linear(50176,4096)
        self.dropout = nn.Dropout()
        self.fc2 = nn.Linear(4096, 7*7*(10 + C))

    def forward(self, x):
        feature = self.features(x)
        output = self.yolo(feature)
        output = self.flatten(output)
        output = F.leaky_relu(self.fc1(output), 0.1)
        output = self.fc2(output)

        return output


class Flatten(nn.Module):
    def __init__(self):
        super(Flatten, self).__init__()

    def forward(self, x):
        return x.view(x.size(0), -1)

