# src/model.py
from transformers import ViTForImageClassification

def create_vit_model(model_name, num_classes):
    """
    Crea un modelo Vision Transformer preentrenado y ajusta la última capa.
    """
    model = ViTForImageClassification.from_pretrained(
        model_name,
        num_labels=num_classes,
        ignore_mismatched_sizes=True  # útil si el head no coincide
    )
    return model
