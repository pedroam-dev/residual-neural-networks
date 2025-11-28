import torch
from torchvision.datasets import ImageFolder
from torchvision.models import ResNet18_Weights
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import torch.optim as optim
import torch.nn as nn
from sys import platform
from torchvision import models
from tqdm import tqdm
import config
from ImageLoader import ImageLoader

if platform == "darwin":
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu") 
else:
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def train(num_epoch, model):
    for epoch in range(0, num_epoch):
        model.train()
        loop = tqdm(enumerate(train_loader), total=len(train_loader)) # create a progress bar
        
        total_Train_Loss = 0
        total_Val_Loss = 0

        train_Correct = 0
        val_Correct = 0

        for batch_idx, (data, targets) in loop:
            data = data.to(device=device)
            targets = targets.to(device=device)
            scores = model(data)
            loss = criterion(scores, targets)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
#            _, preds = torch.max(scores, 1)
#             current_loss += loss.item() * data.size(0)
#             current_corrects += (preds == targets).sum().item()
#             accuracy = int(current_corrects / len(train_loader.dataset) * 100)
            total_Train_Loss += loss
            train_Correct += (scores.argmax(1) == targets).type(torch.float).sum().item()
            loop.set_description(f"Epoch {epoch+1}/{num_epoch} process: {int((batch_idx / len(train_loader)) * 100)}")
            loop.set_postfix(loss=loss.data.item())
        
        with torch.no_grad():
            model.eval()
            for (data, targets) in val_loader:
                (data, targets) = (data.to(device), targets.to(device))
                pred = model (data)
                total_Val_Loss += criterion(pred, targets)
                val_Correct += (pred.argmax(1) == targets).type(torch.float).sum().item()
        
        avg_Train_Loss = total_Train_Loss/train_steps
        avg_Val_Loss = total_Val_Loss/val_steps

        train_Correct = train_Correct/len(train_dataset)
        val_Correct = val_Correct/len(val_dataset)

        H["train_loss"].append(avg_Train_Loss.cpu().detach().numpy())
        H["train_accuracy"].append(train_Correct)
        H["val_loss"].append(avg_Val_Loss.cpu().detach().numpy())
        H["val_accuracy"].append(val_Correct)

     # save model
    torch.save({ 
                'model_state_dict': model.state_dict(), 
                'optimizer_state_dict': optimizer.state_dict(), 
                }, config.PATH_SAVE_MODEL)

def plot_history():
    plt.style.use("ggplot")
    plt.figure()
    plt.plot(H["train_loss"], label="train_loss")
    plt.plot(H["val_loss"], label="val_loss")
    plt.plot(H["train_accuracy"], label="train_accuracy")
    plt.plot(H["val_accuracy"], label="val_accuracy")
    plt.title("Training Loss and Accuracy on Dataset")
    plt.xlabel("Epoch #")
    plt.ylabel("Loss/Accuracy")
    plt.legend(loc="lower left")
    plt.savefig(config.PATH_SAVE_PLOT)



dataset = ImageFolder(config.PATH_DATASET)
train_data, val_data, train_label, val_label = train_test_split(dataset.imgs, dataset.targets, test_size=0.2, random_state=42)

train_transform = transforms.Compose([
    transforms.Resize((224, 224)), transforms.ToTensor(), transforms.Normalize([0.5]*3, [0.5]*3)
]) 

val_transform = transforms.Compose([
    transforms.Resize((224, 224)), transforms.ToTensor(), transforms.Normalize([0.5]*3, [0.5]*3)
]) 

train_dataset = ImageLoader(train_data, train_transform)
val_dataset = ImageLoader(val_data, val_transform)

train_loader = DataLoader(train_dataset, batch_size=config.BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=config.BATCH_SIZE, shuffle=True)

"""for param in model.parameters():
        param.requires_grad = False"""
model = models.resnet18(weights=ResNet18_Weights.DEFAULT)
for module, param in zip(model.modules(), model.parameters()):
	if isinstance(module, nn.BatchNorm2d):
		param.requires_grad = False
num_ftrs = model.fc.in_features
model.fc = nn.Linear(num_ftrs, config.NUM_CLASSES)

model.to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=config.LR)

H = {"train_loss":[], "train_accuracy":[], "val_loss":[],
      "val_accuracy":[]}

train_steps = len(train_dataset) // config.BATCH_SIZE
val_steps = len(val_dataset) // config.BATCH_SIZE




if __name__ == "__main__":
    train(config.TRAIN_EPOCHS, model) 
    plot_history()
