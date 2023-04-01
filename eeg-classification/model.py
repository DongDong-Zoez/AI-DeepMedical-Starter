import torch
import torch.nn as nn

class EEGNet(nn.Module):
    def __init__(self, nb_classes=2, alpha=1.0, p=0.25):
        super(EEGNet, self).__init__()
        
        # Set up the hyperparameters
        self.nb_classes = nb_classes
        self.firstconv = nn.Sequential(
            nn.Conv2d(1, 16, (1, 51), (1, 1), (0, 25), bias=False),
            nn.BatchNorm2d(16)
        )

        self.depthwiseConv = nn.Sequential(
            nn.Conv2d(16, 32, (2, 1), (1, 1), groups=16, bias=False),
            nn.BatchNorm2d(32),
            nn.ELU(alpha),
            nn.AvgPool2d((1, 4), (1, 4)),
            nn.Dropout(p),
        )

        self.separableConv = nn.Sequential(
            nn.Conv2d(32, 32, (1, 15), (1, 1), (0, 7), bias=False),
            nn.BatchNorm2d(32),
            nn.ELU(alpha),
            nn.AvgPool2d((1, 8), (1, 8)),
            nn.Dropout(p),
        )


        self.classify = nn.Sequential(
            nn.Linear(in_features=736, out_features=2)
        )
        
        # Initialize the weights
        self._init_weights()
    
    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.xavier_uniform_(m.weight, gain=1.0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight, gain=1.0)
                nn.init.constant_(m.bias, 0)
    
    def forward(self, x):
        x = self.firstconv(x)
        x = self.depthwiseConv(x)
        x = self.separableConv(x)
        x = x.view(x.size(0), -1)
        x = self.classify(x)    
        return x

if __name__ == "__main__":
    model = EEGNet(nb_classes=2)
    print(model)