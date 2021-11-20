import torch
import torch.nn as nn


# モデルの定義
class Net(nn.Module):
    def __init__(self, classes):
        super().__init__()
        self.conv1 = torch.nn.Sequential(nn.Conv2d(3, 16, 3, 2, 1),
                                         nn.BatchNorm2d(16),
                                         nn.ReLU())
        self.conv2 = torch.nn.Sequential(nn.Conv2d(16, 64, 3, 2, 1),
                                         nn.BatchNorm2d(64),
                                         nn.ReLU())

        self.fc1 = nn.Linear(3136, 100)
        self.dropout = nn.Dropout(0.5)
        self.fc2 = torch.nn.Linear(100, classes)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = x.view(x.size(0), -1)
        x = self.fc1(x)
        x = self.dropout(x)
        x = self.fc2(x)
        return x
