import torch
import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
import os

# Adjustable parameters
image_size = 256
batch_size = 32

# Define dataset paths 
dataset_path = '/kaggle/input/potato-disease-leaf-datasetpld/PLD_3_Classes_256'  # Adjust the path to your dataset

# Define transforms for training (with data augmentation), validation, and testing
train_transform = transforms.Compose([
    transforms.Resize((image_size, image_size)),  # Resize
    transforms.RandomHorizontalFlip(),            # Data augmentation: horizontal flip
    transforms.RandomRotation(15),                # Data augmentation: random rotation
    transforms.ToTensor(),                        # Convert to Tensor
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # Normalize
])

val_test_transform = transforms.Compose([
    transforms.Resize((image_size, image_size)),  # Resize
    transforms.ToTensor(),                       # Convert to Tensor
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # Normalize
])

# Load the dataset using ImageFolder
full_dataset = ImageFolder(os.path.join(dataset_path, 'Training'), transform=train_transform)

# Split the dataset into train, validation, and test sets
train_idx, val_idx = train_test_split(list(range(len(full_dataset))), test_size=0.2, random_state=42)
val_idx, test_idx = train_test_split(val_idx, test_size=0.5, random_state=42)

# Create subsets for training, validation, and test sets
train_set = torch.utils.data.Subset(full_dataset, train_idx)
val_set = torch.utils.data.Subset(full_dataset, val_idx)
test_set = torch.utils.data.Subset(full_dataset, test_idx)

# Update transforms for validation and test sets (no augmentation)
train_set.dataset.transform = train_transform
val_set.dataset.transform = val_test_transform
test_set.dataset.transform = val_test_transform

# Create data loaders
train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False)
test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False)

# Save the loaders for later use
torch.save(train_loader, 'train_loader.pth')
torch.save(val_loader, 'val_loader.pth')
torch.save(test_loader, 'test_loader.pth')

print("Preprocessing with data augmentation completed, and data loaders saved.")
