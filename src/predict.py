# src/predict.py
import torch
from torchvision import transforms
from transformers import ViTConfig, ViTForImageClassification
from PIL import Image
import os
import json

# ======== CONFIGURACIÓN ========
MODEL_PATH = "models/vit_cloud_20251105_111601.pth"  # tu modelo entrenado
IMAGE_PATH = "D:/Tec/Septimo/Sub_periodo_2/Proyecto_Benji/data/test/clear sky/clearsky1.jpg"                 # imagen a clasificar
LABELS_PATH = "data/labels.json"                     # nombres de categorías (opcional)
MODEL_NAME = "google/vit-base-patch16-224"           # el mismo usado en entrenamiento
# =================================

# Cargar etiquetas si existen
if os.path.exists(LABELS_PATH):
    with open(LABELS_PATH, "r", encoding="utf-8") as f:
        class_names = json.load(f)
else:
    # Si no tienes archivo de etiquetas, defínelas manualmente
    class_names = ["cirrus", "cumulus", "altostratus", "nimbostratus", "cumulonimbus", "stratus", "cirrostratus"]

# Cargar modelo
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# Crear el modelo con la misma configuración (sin pesos preentrenados)
config = ViTConfig.from_pretrained("google/vit-base-patch16-224", num_labels=len(class_names))
model = ViTForImageClassification(config)

# Cargar tus pesos entrenados
state_dict = torch.load(MODEL_PATH, map_location=device)
model.load_state_dict(state_dict, strict=True)
model.to(device)
model.eval()

# Preprocesamiento (idéntico al entrenamiento)
preprocess = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5],
                         std=[0.5, 0.5, 0.5])
])

# Cargar y transformar la imagen
image = Image.open(IMAGE_PATH).convert("RGB")
input_tensor = preprocess(image).unsqueeze(0).to(device)

# Inferencia
with torch.no_grad():
    outputs = model(input_tensor)
    probs = torch.nn.functional.softmax(outputs.logits, dim=-1)[0]
    predicted_idx = torch.argmax(probs).item()
    predicted_label = class_names[predicted_idx]

# Mostrar resultado
print(f"\n🔍 Imagen analizada: {os.path.basename(IMAGE_PATH)}")
print(f"☁️  Predicción: {predicted_label}")
print("\n📊 Probabilidades por clase:")
for i, (label, p) in enumerate(zip(class_names, probs)):
    print(f"  {i+1}. {label:<15} {p.item()*100:5.2f}%")
