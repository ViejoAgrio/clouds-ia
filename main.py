# main.py
import torch
from torch import nn, optim
from datetime import datetime
import os
import csv

from src.dataset import get_data_loaders
from src.model import create_vit_model
from src.train import train_one_epoch, validate
from src.config import CONFIG


def log_training_results(cfg, train_loss, train_acc, val_loss, val_acc, val_f1):
    """
    Registra los resultados del entrenamiento en logs/training_log.csv.
    """
    os.makedirs("logs", exist_ok=True)
    log_file = "logs/training_log.csv"
    file_exists = os.path.isfile(log_file)

    with open(log_file, mode="a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        if not file_exists:
            # Encabezado
            writer.writerow([
                "timestamp", "model_name", "epochs", "batch_size", "lr",
                "img_size", "train_loss", "train_acc", "val_loss", "val_acc", "val_f1"
            ])
        # Fila de resultados
        writer.writerow([
            datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            cfg["model_name"],
            cfg["epochs"],
            cfg["batch_size"],
            cfg["learning_rate"],
            cfg["img_size"],
            f"{train_loss:.4f}",
            f"{train_acc:.4f}",
            f"{val_loss:.4f}",
            f"{val_acc:.4f}",
            f"{val_f1:.4f}"
        ])


def main():
    cfg = CONFIG
    device = cfg["device"]

    # 1️⃣ Cargar datos
    train_loader, val_loader, class_names = get_data_loaders(
        cfg["data_dir"], cfg["batch_size"], cfg["img_size"]
    )
    print(f"📂 Clases detectadas: {class_names}")

    # 2️⃣ Modelo ViT
    model = create_vit_model(cfg["model_name"], len(class_names))
    model.to(device)

    # 3️⃣ Pérdida y optimizador
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=cfg["learning_rate"])

    # 4️⃣ Entrenamiento principal
    for epoch in range(cfg["epochs"]):
        print(f"\n🚀 Época {epoch + 1}/{cfg['epochs']}")
        train_loss, train_acc = train_one_epoch(model, train_loader, optimizer, criterion, device)
        val_loss, val_acc, val_f1 = validate(model, val_loader, criterion, device)

        print(f"📊 Resultados -> "
              f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.3f} | "
              f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.3f} | "
              f"Val F1: {val_f1:.3f}")

    # 5️⃣ Guardar modelo
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    os.makedirs("models", exist_ok=True)
    model_path = f"models/vit_cloud_{timestamp}.pth"
    torch.save(model.state_dict(), model_path)
    print(f"✅ Modelo guardado en: {model_path}")

    # 6️⃣ Guardar resumen
    log_training_results(cfg, train_loss, train_acc, val_loss, val_acc, val_f1)
    print("📝 Registro añadido a logs/training_log.csv")


if __name__ == "__main__":
    main()
