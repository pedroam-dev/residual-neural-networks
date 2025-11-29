import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
import os
from PIL import ImageFile
from custom_resnet import CustomResNet

# Para evitar errores con imágenes truncadas
ImageFile.LOAD_TRUNCATED_IMAGES = True

# 1. Configuración y Hiperparámetros
BATCH_SIZE = 32
LEARNING_RATE = 1e-4
EPOCHS = 10

#DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

if torch.backends.mps.is_available():
    DEVICE = torch.device("mps")  # Mac M1/M2/M3
elif torch.cuda.is_available():
    DEVICE = torch.device("cuda")  # NVIDIA GPU
else:
    DEVICE = torch.device("cpu")  # CPU fallback

print(f"Usando dispositivo: {DEVICE}")

DATA_PATH = "/Users/pedroam/Pictures/PetImages" # Ruta del dataset

# 2. Preparación de Datos
transform = transforms.Compose([
    transforms.Resize((128, 128)), # Reducido a 128x128 para agilizar entrenamiento en esta demo
    transforms.ToTensor(),
    transforms.Normalize([0.5]*3, [0.5]*3)
])

# Cargar Dataset (Manejo de errores básico)
try:
    full_dataset = datasets.ImageFolder(DATA_PATH, transform=transform)
    # Split 80/20
    train_size = int(0.8 * len(full_dataset))
    val_size = len(full_dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(full_dataset, [train_size, val_size])

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)
    print(f"Datos cargados: {len(train_dataset)} entrenamiento, {len(val_dataset)} validación.")
except Exception as e:
    print(f"Error cargando dataset: {e}. Verifica la ruta.")
    train_loader = None

# 3. Inicializar Modelo (Usando la clase CustomResNet definida arriba)
# (Asegúrate de copiar la clase CustomResNet aquí o importarla)
model = CustomResNet().to(DEVICE)

# 4. Loss y Optimizador
# Usamos BCELoss porque la salida es Sigmoid (0 a 1)
criterion = nn.BCELoss()
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-4)

# 5. Bucle de Entrenamiento
history = {'train_loss': [], 'val_loss': [], 'val_acc': []}

if train_loader:
    print(f"Iniciando entrenamiento en {DEVICE}...")
    for epoch in range(EPOCHS):
        model.train()
        running_loss = 0.0
        
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(DEVICE), labels.to(DEVICE).float().view(-1, 1)
            
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
        
        # Validación
        model.eval()
        val_loss = 0.0
        correct = 0
        total = 0
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(DEVICE), labels.to(DEVICE).float().view(-1, 1)
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                val_loss += loss.item()
                
                predicted = (outputs > 0.5).float()
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        
        epoch_loss = running_loss / len(train_loader)
        epoch_val_loss = val_loss / len(val_loader)
        epoch_acc = correct / total
        
        history['train_loss'].append(epoch_loss)
        history['val_loss'].append(epoch_val_loss)
        history['val_acc'].append(epoch_acc)
        
        print(f"Epoch {epoch+1}/{EPOCHS} - Loss: {epoch_loss:.4f} - Val Loss: {epoch_val_loss:.4f} - Acc: {epoch_acc:.4f}")

    # 6. Graficar Resultados
    plt.figure(figsize=(12, 5))
    
    # Gráfica de Loss
    plt.subplot(1, 2, 1)
    plt.plot(history['train_loss'], label='Train Loss')
    plt.plot(history['val_loss'], label='Val Loss')
    plt.title('Curvas de Pérdida (Loss)')
    plt.xlabel('Época')
    plt.ylabel('Loss')
    plt.legend()
    
    # Gráfica de Accuracy
    plt.subplot(1, 2, 2)
    plt.plot(history['val_acc'], label='Val Accuracy', color='green')
    plt.title('Precisión de Validación (Accuracy)')
    plt.xlabel('Época')
    plt.ylabel('Accuracy')
    plt.legend()
    
    plt.tight_layout()
    plt.show()