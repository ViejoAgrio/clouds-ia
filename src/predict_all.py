import torch
from torchvision import transforms
from transformers import ViTConfig, ViTForImageClassification
from PIL import Image
import os
import json
from sklearn.metrics import f1_score, confusion_matrix, classification_report
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# ========= CONFIGURACIÓN =========
MODEL_PATH = "models\\vit_cloud_20251117_190341.pth"   # modelo entrenado
TEST_DIR = "Data/test"                                # carpeta test
LABELS_PATH = "data/labels.json"                      # etiquetas opcionales
MODEL_NAME = "google/vit-base-patch16-224"            # igual al entrenamiento
# =================================

# Cargar etiquetas
with open(LABELS_PATH, "r", encoding="utf-8") as f:
    class_names = json.load(f)

# ========= CARGAR MODELO =========
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
config = ViTConfig.from_pretrained(MODEL_NAME, num_labels=len(class_names))
model = ViTForImageClassification(config)

state_dict = torch.load(MODEL_PATH, map_location=device)
model.load_state_dict(state_dict, strict=True)
model.to(device)
model.eval()

# ========= TRANSFORMACIONES =========
preprocess = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5],
                         std=[0.5, 0.5, 0.5])
])

# ========= PREDICCIÓN EN LOTE =========
y_true, y_pred = [], []

print("\n🚀 Iniciando inferencia sobre carpeta test...\n")

for cluster in sorted(os.listdir(TEST_DIR)):
    cluster_path = os.path.join(TEST_DIR, cluster)
    if not os.path.isdir(cluster_path):
        continue

    for img_name in os.listdir(cluster_path):
        if not img_name.lower().endswith((".png", ".jpg", ".jpeg")):
            continue

        img_path = os.path.join(cluster_path, img_name)
        try:
            image = Image.open(img_path).convert("RGB")
            input_tensor = preprocess(image).unsqueeze(0).to(device)

            with torch.no_grad():
                outputs = model(input_tensor)
                probs = torch.nn.functional.softmax(outputs.logits, dim=-1)[0]
                pred_idx = torch.argmax(probs).item()

            y_true.append(class_names.index(cluster))
            y_pred.append(pred_idx)

        except Exception as e:
            print(f"⚠️ Error procesando {img_path}: {e}")

# ========= MÉTRICAS =========
print("\n📊 RESULTADOS GENERALES:\n")

print(classification_report(
    y_true, y_pred,
    target_names=class_names,
    digits=3,
    zero_division=0
))

# F1 macro, micro, weighted
f1_macro = f1_score(y_true, y_pred, average="macro")
f1_micro = f1_score(y_true, y_pred, average="micro")
f1_weighted = f1_score(y_true, y_pred, average="weighted")

print(f"🔹 F1-macro:    {f1_macro:.4f}")
print(f"🔹 F1-micro:    {f1_micro:.4f}")
print(f"🔹 F1-weighted: {f1_weighted:.4f}")

# ========= MATRIZ DE CONFUSIÓN =========
cm = confusion_matrix(y_true, y_pred, labels=range(len(class_names)))
cm_norm = cm.astype("float") / cm.sum(axis=1, keepdims=True)

plt.figure(figsize=(12, 10))
sns.heatmap(cm_norm, annot=False, cmap="Blues",
            xticklabels=class_names, yticklabels=class_names)
plt.title("Matriz de Confusión Normalizada")
plt.xlabel("Predicción")
plt.ylabel("Etiqueta Real")
plt.tight_layout()
plt.show()

