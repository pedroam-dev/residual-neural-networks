from sys import platform
import torchvision.transforms as transforms
from torchvision import models
import torch.nn as nn
from PIL import Image
from io import BytesIO
import torch

def load_model(path_model):
    checkpoint = torch.load(path_model, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])

def prediction(path_image):
    model.eval()

    image = Image.open(BytesIO(path_image))
    image = transform(image)
    image = torch.unsqueeze(image, 0)

    with torch.no_grad():
        outputs = model(image.to(device))

    output_label = torch.topk(outputs, 1)
    pred_class = labels[int(output_label.indices)]
    return pred_class

if platform == "darwin":
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu") 
else:
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

labels = ["Cat", "Dog"]
transform = transforms.Compose([transforms.Resize((224, 224)), transforms.ToTensor(), transforms.Normalize([0.5]*3, [0.5]*3)])

model = models.resnet18(weights=None)
for param in model.parameters():
    param.requires_grad = False
num_ftrs = model.fc.in_features
model.fc = nn.Linear(num_ftrs, 2)
model.to(device)

