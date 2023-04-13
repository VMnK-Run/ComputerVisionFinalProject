
import torch.nn as nn
from torchvision.models import resnet34
import torch.nn.functional as F



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
