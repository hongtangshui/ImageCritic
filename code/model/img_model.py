import torch
import torch.nn as nn
import torch.nn.functional as F

from torchvision import models

class ResNet50(nn.Module):
    '''
    ResNet50 as image encoder
    '''
    def __init__(self, output_dim, pretrained=False):
        super(ResNet50, self).__init__()
        if pretrained:
            self.resnet50 = models.resnet50(weights='DEFAULT')
        else:
            self.resnet50 = models.resnet50()
        self.in_features = self.resnet50.fc.in_features
        self.resnet50.fc = nn.Sequential(
            nn.Linear(self.in_features, output_dim*2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(output_dim*2, output_dim),
        )

    def forward(self, x):
        '''
        Input:
            X : [batch_size, 3, size, size]
            y : [batch_size, output_dim]
        '''
        x = self.resnet50(x)
        return x


class ViT(nn.Module):
    def __init__(self, ):
        super(ViT, self).__init__()

    def forward(self, x):
        pass


if __name__ == "__main__":
    x = torch.zeros((32, 3, 224, 224))
    resnet50 = ResNet50(output_dim=64)
    y = resnet50(x)
    print(y.shape)