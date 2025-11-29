# Cats and Dogs Classification 🐱🐶

Proyecto de clasificación de imágenes de gatos y perros usando Redes Neuronales Residuales (ResNet) con PyTorch.

## Descripción

Este proyecto implementa dos enfoques para clasificar imágenes de gatos y perros:

1. **Transfer Learning con ResNet18** (`train_model.py`) - Usa un modelo preentrenado de ResNet18
2. **ResNet personalizada** (`train_custom.py`) - Implementación desde cero de una arquitectura ResNet

## Configuración del entorno

### Requisitos Previos
- Python 3.10+
- Conda (Anaconda o Miniconda)
- Mac con chip M1/M2/M3 o GPU NVIDIA (opcional, también funciona en CPU)

### Instalación

1. **Clonar el repositorio**
```bash
git clone https://github.com/pedroam-dev/residual-neural-networks.git
cd residual-neural-networks
```

2. **Crear entorno virtual con Conda**
```bash
conda create -n residual-nn python=3.10 -y
conda activate residual-nn
```

3. **Instalar dependencias**
```bash
pip install -r requirements.txt
```

### Dataset

Descarga el dataset de Kaggle: [Dogs vs. Cats](https://www.kaggle.com/c/dogs-vs-cats/data)

Estructura esperada:
```
data/kagglecatsanddogs_5340/PetImages/
├── Cat/
│   ├── 0.jpg
│   ├── 1.jpg
│   └── ...
└── Dog/
    ├── 0.jpg
    ├── 1.jpg
    └── ...
```

**Importante:** El dataset puede contener imágenes corruptas. Usa el script de limpieza:

```bash
python src/models/clean_dataset.py
```

## Uso

### 1. Entrenar modelo con Transfer Learning (ResNet18)
```bash
cd src/models
python train_model.py
```

Este script:
- Carga ResNet18 preentrenado
- Ajusta la última capa para 2 clases (Cat/Dog)
- Guarda el modelo en `models/cat-dogs-model.pt`
- Genera gráficos de entrenamiento en `reports/`

### 2. Entrenar modelo ResNet personalizado
```bash
cd src/models
python train_custom.py
```

Este script:
- Usa una implementación personalizada de ResNet desde cero
- Entrena con bloques residuales
- Optimizado para Mac M1/M2/M3 (MPS) y CUDA

### 3. Hacer predicciones
```bash
cd src/models
python predict_model.py
```

### 4. API REST con FastAPI (opcional)
```bash
cd src/apis
uvicorn cats_dogs_classification_fastapi:app --reload
```

Accede a la API en: `http://localhost:8000/docs`

## Estructura del Proyecto

```
residual-neural-networks/
├── data/                           # Datasets
│   └── test_images/               # Imágenes de prueba
├── models/                         # Modelos entrenados (.pt)
├── reports/                        # Gráficos y reportes
├── src/
│   ├── apis/                      # APIs REST
│   │   ├── cats_dogs_classification_fastapi.py
│   │   └── predict.py
│   └── models/                    # Scripts de entrenamiento
│       ├── clean_dataset.py       # Limpieza de imágenes corruptas
│       ├── config.py              # Configuración
│       ├── custom_resnet.py       # Arquitectura ResNet personalizada
│       ├── ImageLoader.py         # Cargador de imágenes
│       ├── model.py               # Definición del modelo
│       ├── predict_model.py       # Inferencia
│       ├── train_custom.py        # Entrenamiento ResNet custom
│       └── train_model.py         # Entrenamiento Transfer Learning
├── requirements.txt               # Dependencias
└── README.md
```

## Configuración

### Hiperparámetros (train_custom.py)
```python
BATCH_SIZE = 32
LEARNING_RATE = 1e-4
EPOCHS = 10
IMAGE_SIZE = 128x128
```

### Dispositivo (GPU/CPU)
El código detecta automáticamente:
- **Mac M1/M2/M3**: Usa MPS (Metal Performance Shaders)
- **NVIDIA GPU**: Usa CUDA
- **Fallback**: CPU

## Resultados

El entrenamiento genera:
- Modelo guardado: `models/cat-dogs-model.pt`
- Gráficos de pérdida y precisión: `reports/training_plot.png`
- Logs de entrenamiento en consola

## Solución de Problemas

### Error: "cannot identify image file"
Ejecuta el script de limpieza del dataset:
```bash
python src/models/clean_dataset.py
```

### Error: "No module named 'torch'"
Verifica que el entorno conda esté activado:
```bash
conda activate residual-nn
pip install -r requirements.txt
```

### Versión de Python incompatible
Usa Python 3.10-3.12:
```bash
conda create -n residual-nn python=3.10 -y
```

## Requisitos del sistema

- **RAM**: 8GB mínimo (16GB recomendado)
- **Espacio**: ~5GB para dataset + modelos
- **GPU**: Opcional pero recomendado para entrenamiento más rápido

## Contribuciones

Las contribuciones son bienvenidas. Por favor:
1. Fork el proyecto
2. Crea una rama para tu feature (`git checkout -b feature/AmazingFeature`)
3. Commit tus cambios (`git commit -m 'Add some AmazingFeature'`)
4. Push a la rama (`git push origin feature/AmazingFeature`)
5. Abre un Pull Request

## Autor

**Pedro AM**
- GitHub: [@pedroam-dev](https://github.com/pedroam-dev)

## Agradecimientos

- Dataset: [Kaggle Dogs vs. Cats](https://www.kaggle.com/c/dogs-vs-cats)
- PyTorch por el framework
