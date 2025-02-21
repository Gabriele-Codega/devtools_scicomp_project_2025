import torch.nn as nn

class AlexNet(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.num_classes = num_classes
        self.features = nn.Sequential(
            # here insert convolutional blocks
            nn.Conv2d(in_channels=3,out_channels=96,kernel_size=11,stride=1,padding=2),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(in_channels=96,out_channels=256,kernel_size=5,stride=1,groups=2),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(in_channels=256,out_channels=384,kernel_size=3,stride=1,groups=2),
            nn.ReLU(),
            nn.Conv2d(in_channels=384,out_channels=384,kernel_size=3,stride=1,groups=2),
            nn.ReLU(),
            nn.Conv2d(in_channels=384,out_channels=256,kernel_size=3,stride=1,groups=2),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2)
        )
        self.avgpool = nn.AdaptiveAvgPool2d((6, 6))
        self.classifier = nn.Sequential(
            # here insert the linear + dropout blocks
            nn.Dropout(0.5),
            nn.Linear(256*6*6,4096),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(4096,4096),
            nn.ReLU(),
            nn.Linear(4096, self.num_classes),
        )
    def forward(self, x):
        x = self.avgpool(self.features(x)).flatten(start_dim=1)
        logits = self.classifier(x)
        return logits
