"""
dataset_loader.py
Loads slot patch images and applies transforms/augmentations for model training & evaluation.
"""

import os
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

PATCHES_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "patches")

# ImageNet normalization statistics
NORM_MEAN = [0.485, 0.456, 0.406]
NORM_STD = [0.229, 0.224, 0.225]

def get_data_transforms():
    train_transforms = transforms.Compose([
        transforms.Resize((64, 64)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ColorJitter(brightness=0.2, contrast=0.2),
        transforms.ToTensor(),
        transforms.Normalize(mean=NORM_MEAN, std=NORM_STD)
    ])
    
    val_test_transforms = transforms.Compose([
        transforms.Resize((64, 64)),
        transforms.ToTensor(),
        transforms.Normalize(mean=NORM_MEAN, std=NORM_STD)
    ])
    
    return train_transforms, val_test_transforms

def get_dataloaders(batch_size=64, num_workers=0):
    train_tf, val_tf = get_data_transforms()
    
    train_dir = os.path.join(PATCHES_DIR, "train")
    val_dir = os.path.join(PATCHES_DIR, "valid")
    test_dir = os.path.join(PATCHES_DIR, "test")
    
    train_dataset = datasets.ImageFolder(train_dir, transform=train_tf)
    val_dataset = datasets.ImageFolder(val_dir, transform=val_tf)
    test_dataset = datasets.ImageFolder(test_dir, transform=val_tf)
    
    class_names = train_dataset.classes
    print(f"Dataset classes found: {class_names} (mapping: {train_dataset.class_to_idx})")
    print(f"Samples: Train={len(train_dataset)}, Valid={len(val_dataset)}, Test={len(test_dataset)}")
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    
    return train_loader, val_loader, test_loader, class_names, train_dataset.class_to_idx

if __name__ == "__main__":
    t_loader, v_loader, te_loader, classes, mapping = get_dataloaders(batch_size=32)
    print("DataLoader test passed successfully!")
