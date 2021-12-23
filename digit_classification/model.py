import torch
from torch import nn
import torchvision.models as models
from tqdm import tqdm

def get_model(model_name):
    if model_name == "cnn":
        return CNN()
    else:
        raise ValueError("please input model name")

# ------------ classification model
class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, padding=1), 
            nn.BatchNorm2d(16),
            nn.Dropout(0.2),  
            nn.ReLU(),
            nn.MaxPool2d(2)  
        )

        self.layer2 = nn.Sequential(
            nn.Conv2d(16, 64, kernel_size=3, padding=1), 
            nn.BatchNorm2d(64),
            nn.Dropout(0.2),  
            nn.ReLU(),
            nn.MaxPool2d(2) 
        )

        self.layer3 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, padding=1), 
            nn.BatchNorm2d(128),
            nn.Dropout(0.2),  
            nn.ReLU(),
            nn.MaxPool2d(2)  
        )
            
        self.fc1 = nn.Sequential(
            nn.Linear(1152, 1024),
            nn.Dropout(0.2),  
            nn.ReLU()
        )

        self.fc2 = nn.Linear(1024, 10)

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
#         x = self.layer4(x)
        x = x.view(x.size(0), -1)
        x = self.fc1(x)
        x = self.fc2(x)
        return x
        
