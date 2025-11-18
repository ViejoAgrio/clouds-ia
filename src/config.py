# src/config.py
import torch

CONFIG = {
    "data_dir": "./data", 
    "batch_size": 16,
    "img_size": 224,
    "epochs": 20,
    "learning_rate": 1e-4,
    "device": "cuda" if torch.cuda.is_available() else "cpu",
    "model_name": "google/vit-base-patch16-224",
}
