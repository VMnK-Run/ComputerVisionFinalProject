import torch
import torch.nn as nn
import torch.nn.functional as F


class VGG19(nn.Module):
    def __init__(self):
        super(VGG19, self).__init__()
        self.in_channels = 3
        self.conv3_64 = self.__make_layer(64, 2)
        self.conv3_128 = self.__make_layer(128, 2)
        self.conv3_256 = self.__make_layer(256, 4)
        self.conv3_512a = self.__make_layer(512, 4)
        self.conv3_512b = self.__make_layer(512, 4)
        self.fc = nn.Linear(512, 7)

    def forward(self, x, labels=None):
        out = self.conv3_64(x)
        out = F.max_pool2d(out, 2)
        out = self.conv3_128(out)
        out = F.max_pool2d(out, 2)
        out = self.conv3_256(out)
        out = F.max_pool2d(out, 2)
        out = self.conv3_512a(out)
        out = F.max_pool2d(out, 2)
        out = self.conv3_512b(out)
        out = F.max_pool2d(out, 2)
        out = out.view(out.size(0), -1)
        out = F.dropout(out, p=0.5, training=self.training)
        out = self.fc(out)
        if labels is not None:
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(out, labels)
            return loss, out
        else:
            return out

    def __make_layer(self, channels, num):
        layers = []
        for i in range(num):
            layers.append(nn.Conv2d(self.in_channels, channels, kernel_size=3, stride=1, padding=1))
            layers.append(nn.BatchNorm2d(channels))
            layers.append(nn.ReLU())
            self.in_channels = channels
        return nn.Sequential(*layers)


class BasicBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)

        self.extra = nn.Sequential()
        # 保证降尺寸和升维时可以相加
        if stride != 1 or in_channels != out_channels:
            self.extra = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride),
                nn.BatchNorm2d(out_channels)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.extra(x)
        out = F.relu(out)
        return out


class ResNet18(nn.Module):
    def __init__(self, num_classes=7):
        super(ResNet18, self).__init__()
        self.in_channels = 64

        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(64)
        self.conv2 = self.__make_layer(64, 2, 1)
        self.conv3 = self.__make_layer(128, 2, 2)
        self.conv4 = self.__make_layer(256, 2, 2)
        self.conv5 = self.__make_layer(512, 2, 2)
        self.linear = nn.Linear(512, num_classes)

    def forward(self, x, labels=None):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.conv2(out)
        out = self.conv3(out)
        out = self.conv4(out)
        out = self.conv5(out)
        out = F.avg_pool2d(out, 4)
        out = out.view(out.size(0), -1)
        out = F.dropout(out, p=0.5, training=self.training)
        out = self.linear(out)
        if labels is not None:
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(out, labels)
            return loss, out
        else:
            return out

    def __make_layer(self, channels, nums, stride):
        strides = [stride] + [1] * (nums - 1)
        layers = []
        for i in range(nums):
            stride = strides[i]
            layers.append(BasicBlock(self.in_channels, channels, stride))
            self.in_channels = channels
        return nn.Sequential(*layers)
