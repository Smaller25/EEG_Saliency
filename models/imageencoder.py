import torch
import torch.nn as nn
from torchvision.models import resnet18



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
        self.inception = torch.hub.load('pytorch/vision:v0.10.0', 'inception_v3', weights='DEFAULT', aux_logits=True)
        # fc layer 변경 필요
        in_features = self.inception.fc.in_features
        self.inception.fc = nn.Linear(in_features, out_dim)
    
    def forward(self, x):
        out = self.inception(x)
        if isinstance(out, tuple):
            out = out[0]
        return out