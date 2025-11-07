# src/config.py
import torch

CONFIG = {
    "data_dir": "./data",         # ruta a tus imágenes
    "batch_size": 16,
    "img_size": 224,
    "epochs": 25,
    "learning_rate": 1e-4,
    "device": "cuda" if torch.cuda.is_available() else "cpu",
    "model_name": "google/vit-base-patch16-224",
    "num_classes": 7
}
