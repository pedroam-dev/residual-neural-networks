from torchvision import models
import torch.nn as nn 

class Net():
    def __init__(self):
        model = models.resnet50(pretrained=True)
        for param in model.parameters():
            param.requires_grad = False

        num_ftrs = model.fc.in_features
        model.fc = nn.Linear(num_ftrs, 2)
        return model