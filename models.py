import torch
import torch.nn as nn
from torchvision.models import resnet18


# 일단은 leaky ReLU 버전으로 함 
# 참고 : https://github.com/secondlevel/EEG-classification/blob/main/EEGNet_training_LeakyReLU.py

class EEGNet(nn.Module):
    def __init__(self, chans=128, time_points=440,   # input size (batch_size, channels, time_point)
                 feature_dim=128,                       # output feature dimension
                 temp_kernel=32, f1=16, f2=32, d=2,     # kernel size, feature_dim size
                 pk1=8, pk2=8,                          # pooling kernel size
                 dropout_rate=0.5):                     # dropout_rate
        super(EEGNet, self).__init__()

        linear_size = (time_points //(pk1 * pk2)) * f2

        self.firstConv = nn.Sequential(
            nn.Conv2d(1, f1, kernel_size=(1, temp_kernel), padding='same', bias=False),
            nn.BatchNorm2d(f1)
        )

        self.depthwiseConv = nn.Sequential(
            nn.Conv2d(f1, d * f1, (chans, 1), groups=f1, bias=False),
            nn.BatchNorm2d(d * f1),
            nn.ELU(),
            nn.AvgPool2d((1, pk1)),
            nn.Dropout(dropout_rate)
        )
        self.separableConv = nn.Sequential(
            nn.Conv2d(d * f1, f2, (1, 16), groups=f2, bias=False, padding='same'),
            nn.Conv2d(f2, f2, kernel_size=1, bias=False),
            nn.BatchNorm2d(f2),
            nn.ELU(),
            nn.AvgPool2d((1, pk2)),
            nn.Dropout(dropout_rate)
        )
        self.flatten = nn.Flatten()
        self.fc = nn.Linear(linear_size, feature_dim)

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
        self.inception = torch.hub.load('pytorch/vision:v0.10.0', 'inception_v3', weights='DEFAULT', aux_logits=True)
        # fc layer 변경 필요
        in_features = self.inception.fc.in_features
        self.inception.fc = nn.Linear(in_features, out_dim)
    
    def forward(self, x):
        out = self.inception(x)
        if isinstance(out, tuple):
            out = out[0]
        return out
    

# import torchvision.models as models
# import torch.nn as nn

# class Inception_ImageEncoder(nn.Module):
#     def __init__(self, out_dim=64):
#         super().__init__()
#         # torchvision을 통해 직접 불러오기
#         self.inception = models.inception_v3(pretrained=True, aux_logits=False)
        
#         # fc 레이어 수정
#         in_features = self.inception.fc.in_features
#         self.inception.fc = nn.Linear(in_features, out_dim)

#     def forward(self, x):
#         return self.inception(x)