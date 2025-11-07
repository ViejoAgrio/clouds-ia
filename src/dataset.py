# src/dataset.py
from torchvision import transforms, datasets
from torch.utils.data import DataLoader

def get_data_loaders(data_dir, batch_size=16, img_size=224):
    """
    Crea los dataloaders con augmentación para entrenamiento y validación.
    """

    # Aumentos fuertes pero realistas para cielos/nubes
    train_transforms = transforms.Compose([
        transforms.RandomResizedCrop(img_size, scale=(0.8, 1.0)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomVerticalFlip(p=0.3),
        transforms.RandomRotation(degrees=45),
        transforms.ColorJitter(
            brightness=0.4,    # variación de iluminación
            contrast=0.4,
            saturation=0.4,
            hue=0.1
        ),
        transforms.RandomAffine(
            degrees=0,
            translate=(0.1, 0.1),  # pequeños desplazamientos
            shear=10
        ),
        transforms.RandomPerspective(distortion_scale=0.2, p=0.3),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],  # valores estándar de ImageNet
            std=[0.229, 0.224, 0.225]
        )
    ])

    # Validación sin aumentos, solo redimensionado y normalización
    val_transforms = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])

    train_dataset = datasets.ImageFolder(root=f"{data_dir}/train", transform=train_transforms)
    val_dataset   = datasets.ImageFolder(root=f"{data_dir}/val", transform=val_transforms)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
    val_loader   = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4)

    return train_loader, val_loader, train_dataset.classes
