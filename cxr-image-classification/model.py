import timm
import torch
import torch.nn as nn
from collections import OrderedDict

# Third-party packages


class CXRNet(nn.Module):

    def __init__(self, model_name, num_classes, hidden_size=384):
        super().__init__()

        self.features = timm.create_model(
            model_name, num_classes=num_classes, pretrained=True)

        for param in self.features.parameters():
            param.requires_grad = False

        # Replace the last layer with a new fully connected layer with the correct number of classes
        self.features.classifier = nn.Sequential(
            OrderedDict([
                ('fcl1', nn.Linear(self.features.classifier.in_features, 256)),
                ('dp1', nn.Dropout(0.3)),
                ('r1', nn.ReLU()),
                ('fcl2', nn.Linear(256, 32)),
                ('dp2', nn.Dropout(0.3)),
                ('r2', nn.ReLU()),
                ('fcl3', nn.Linear(32, num_classes)),
                ('out', nn.LogSoftmax(dim=1)),
            ])
        )

    def forward(self, x):
        return self.features(x)
