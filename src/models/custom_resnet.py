import torch
import torch.nn as nn
import torch.nn.functional as F

# ---------------------------------------------------------
# Bloques Auxiliares basados en el Diagrama
# ---------------------------------------------------------

class ConvBatchNormRelu(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding):
        super(ConvBatchNormRelu, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, bias=False)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.relu(self.bn(self.conv(x)))

class IdentityBlock(nn.Module):
    """
    Bloque Identity según diagrama.
    Nota: El diagrama indica s:2 en la conv, pero un shortcut directo.
    Se usa s=1 para que las dimensiones coincidan para la suma.
    """
    def __init__(self, f):
        super(IdentityBlock, self).__init__()
        # Rama principal
        self.conv1 = ConvBatchNormRelu(f, f, kernel_size=3, stride=1, padding=1)
        self.conv2 = ConvBatchNormRelu(f, f, kernel_size=3, stride=1, padding=1)
        # La última ReLU va después de la suma
        self.final_relu = nn.ReLU(inplace=True)

    def forward(self, x):
        shortcut = x
        out = self.conv1(x)
        # Hack para acceder a conv+bn sin relu en la segunda etapa (como es usual en ResNet)
        out = self.conv2.conv(out)
        out = self.conv2.bn(out)
        out += shortcut
        return self.final_relu(out)

class ProjectionBlock(nn.Module):
    """
    Bloque Projection según diagrama.
    Aumenta canales o reduce dimensiones.
    Diagrama: Conv1 s:2, Conv2 s:1. Shortcut: Conv 1x1 s:2.
    """
    def __init__(self, in_f, out_f, stride=2):
        super(ProjectionBlock, self).__init__()
        # Rama principal
        self.conv1 = ConvBatchNormRelu(in_f, out_f, kernel_size=3, stride=stride, padding=1)
        self.conv2 = ConvBatchNormRelu(out_f, out_f, kernel_size=3, stride=1, padding=1)
        
        # Rama Shortcut (Proyección)
        self.shortcut_conv = nn.Conv2d(in_f, out_f, kernel_size=1, stride=stride, bias=False)
        self.shortcut_bn = nn.BatchNorm2d(out_f)
        
        self.final_relu = nn.ReLU(inplace=True)

    def forward(self, x):
        shortcut = self.shortcut_bn(self.shortcut_conv(x))
        
        out = self.conv1(x)
        # Acceso manual a conv+bn sin relu para la suma final
        out = self.conv2.conv(out)
        out = self.conv2.bn(out)
        
        out += shortcut
        return self.final_relu(out)

# ---------------------------------------------------------
# Arquitectura Principal CustomResNet
# ---------------------------------------------------------

class CustomResNet(nn.Module):
    def __init__(self):
        super(CustomResNet, self).__init__()
        
        # --- Capas Iniciales ---
        # f: 16, k: 7, s: 1 (Padding 3 para mantener tamaño si fuera necesario, o reducir un poco)
        # El diagrama dice s:1.
        self.initial_conv = ConvBatchNormRelu(3, 16, kernel_size=7, stride=1, padding=3)
        
        # MaxPool ps: 3, s: 1 (Padding 1 para mantener tamaño)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=1, padding=1)

        # --- ResNet 1 (f: 16) ---
        # Composición según diagrama: Identity -> Identity
        self.resnet_1 = nn.Sequential(
            IdentityBlock(16),
            IdentityBlock(16)
        )

        # --- ResNet 2 (f: 32) ---
        # Composición según diagrama: Projection -> Identity
        self.resnet_2 = nn.Sequential(
            ProjectionBlock(16, 32),
            IdentityBlock(32)
        )

        # --- ResNet 3 Primera Pasada (f: 64) ---
        # Composición según diagrama: Identity -> Projection
        # Nota: Entramos con 32. Identity(32) -> Projection(32->64)
        self.resnet_3_a = nn.Sequential(
            IdentityBlock(32),
            ProjectionBlock(32, 64)
        )

        # --- ResNet 3 Segunda Pasada (f: 128) ---
        # El diagrama muestra otro bloque resnet_3 debajo.
        # Entramos con 64. Identity(64) -> Projection(64->128)
        self.resnet_3_b = nn.Sequential(
            IdentityBlock(64),
            ProjectionBlock(64, 128)
        )

        # --- Clasificador ---
        self.global_avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        # FC u:1
        self.fc = nn.Linear(128, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.initial_conv(x)
        x = self.maxpool(x)
        
        x = self.resnet_1(x)
        x = self.resnet_2(x)
        x = self.resnet_3_a(x)
        x = self.resnet_3_b(x)
        
        x = self.global_avg_pool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        x = self.sigmoid(x)
        return x

# Verificación rápida del modelo
if __name__ == "__main__":
    model = CustomResNet()
    # Prueba con un tensor aleatorio para verificar dimensiones
    dummy_input = torch.randn(1, 3, 224, 224)
    output = model(dummy_input)
    print(f"Output shape: {output.shape}") # Debería ser [1, 1]