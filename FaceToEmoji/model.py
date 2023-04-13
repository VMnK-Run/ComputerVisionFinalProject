import torch
import torch.nn as nn
from torchvision.models import resnet18
from torchvision.models import resnet34
import torch.nn.functional as F









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


class EmojiResNet18(nn.Module):
    def __init__(self, num_classes=7):
        super(EmojiResNet18, self).__init__()
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
    






class FaceClassifierResNet18(nn.Module):
    def __init__(self, pretrained=True):
        super(FaceClassifierResNet18, self).__init__()
        
        self.base_model = resnet18(pretrained=pretrained)
        num_features = self.base_model.fc.in_features
        
        # 替换最后的全连接层
        self.base_model.fc = nn.Linear(num_features, 1)

    def forward(self, x):
        x = self.base_model(x)
        x = torch.sigmoid(x)  # 将输出值范围限定在0和1之间
        return x # 删除输出张量中的额外维度



class FaceDetectorResNet34(nn.Module):
    def __init__(self):
        super(FaceDetectorResNet34, self).__init__()
        resnet = resnet34(pretrained=True)
        
        # 移除ResNet的最后一个全连接层
        self.resnet = nn.Sequential(*list(resnet.children())[:-1])
        
        # 用于回归的全连接层
        self.fc = nn.Linear(resnet.fc.in_features, 4)

    def forward(self, x):

        x = self.resnet(x)
        
        # 将特征展平
        x = x.view(x.size(0), -1)
        
        # 应用全连接层进行回归
        x = self.fc(x)
        return x
