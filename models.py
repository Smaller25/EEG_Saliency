import torch
import torch.nn as nn
from torchvision.models import resnet18


# 일단은 leaky ReLU 버전으로 함 
# 참고 : https://github.com/secondlevel/EEG-classification/blob/main/EEGNet_training_LeakyReLU.py

class EEGNet(nn.Module):
    def __init__(self, feature_dim=128):
        super(EEGNet, self).__init__()

        self.firstConv = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=(1, 51), stride=(1, 1), padding=(0, 25), bias=False),
            nn.BatchNorm2d(16)
        )

        self.depthwiseConv = nn.Sequential(
            nn.Conv2d(16, 32, kernel_size=(2, 1), stride=(1, 1), groups=8, bias=False),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(0.06),
            nn.AvgPool2d(kernel_size=(1, 4), stride=(1, 4)),
            nn.Dropout(0.5)
        )

        self.separableConv = nn.Sequential(
            nn.Conv2d(32, 32, kernel_size=(1, 15), stride=(1, 1), padding=(0, 7), bias=False),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(0.06),
            nn.AvgPool2d(kernel_size=(1, 8), stride=(1, 8)),
            nn.Dropout(0.5)
        )

        self.flatten = nn.Flatten()
        self.fc = nn.Linear(736, feature_dim)  # output 736 units from conv, to desired feature dim

    def forward(self, x):
        x = self.firstConv(x)
        x = self.depthwiseConv(x)
        x = self.separableConv(x)
        x = self.flatten(x)
        x = self.fc(x)
        return x



class ResNet_ImageEncoder(nn.Module):
    def __init__(self, out_dim=64):
        super().__init__()
        self.resnet = resnet18(pretrained=True)
        self.resnet.fc = nn.Linear(512, out_dim)

    def forward(self, x):
        with torch.no_grad():
            return self.resnet(x)
        


# 참고 : https://pytorch.org/hub/pytorch_vision_inception_v3/
class Inception_ImageEncoder(nn.Module):
    def __init__(self, out_dim = 64):
        super().__init__()
        self.inception = torch.hub.load('pytorch/vision:v0.10.0', 'inception_v3', pretrained=True)
        self.inception.fc = nn.Linear(512, out_dim)
    
    def forward(self, x):
        with torch.no_grad():
            return self.inception