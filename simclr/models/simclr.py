import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models.resnet import resnet50

class SimClr(nn.Module):
    def __init__(self, opt):
        super(SimClr, self).__init__()
        self.encoder = []
        for name, module in resnet50().named_children():
            if name == 'conv1':
                module = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
            if not isinstance(module, nn.Linear) and not isinstance(module, nn.MaxPool2d):
                self.encoder.append(module)
        self.encoder = nn.Sequential(*self.encoder)
        self.project = nn.Sequential(nn.Linear(2048, 512, bias=False), nn.BatchNorm1d(512),
                               nn.ReLU(inplace=True), nn.Linear(512, opt['feature_dim'], bias=True))


    def forward(self, x):
      x = self.encoder(x)
      feature = torch.flatten(x, start_dim=1)
      out = self.project(feature)
      return F.normalize(feature, dim=-1), F.normalize(out, dim=-1)


