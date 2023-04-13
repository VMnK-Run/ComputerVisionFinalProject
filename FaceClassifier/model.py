import torch
import torch.nn as nn
from torchvision.models import resnet18
from torchvision.models import resnet34
import torch.nn.functional as F


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