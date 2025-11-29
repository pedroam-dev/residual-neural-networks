import os
from PIL import Image

def clean_dataset(data_path):
    """Elimina imágenes corruptas del dataset"""
    removed = 0
    for root, dirs, files in os.walk(data_path):
        for file in files:
            if file.endswith(('.jpg', '.jpeg', '.png')):
                file_path = os.path.join(root, file)
                try:
                    img = Image.open(file_path)
                    img.verify()  # Verifica que la imagen sea válida
                    img.close()
                except Exception as e:
                    print(f"Eliminando archivo corrupto: {file_path}")
                    os.remove(file_path)
                    removed += 1
    print(f"Total de archivos eliminados: {removed}")

if __name__ == "__main__":
    DATA_PATH = "/Users/pedroam/Pictures/PetImages/"
    clean_dataset(DATA_PATH)