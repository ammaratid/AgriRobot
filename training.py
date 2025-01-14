import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from torchvision import models

# Define the custom model architecture
class CustomModel(nn.Module):
    def __init__(self, num_classes=3):
        super(CustomModel, self).__init__()
        self.mobilenet = models.mobilenet_v2(pretrained=True)
        self.mobilenet.classifier[1] = nn.Linear(self.mobilenet.last_channel, num_classes)

    def forward(self, x):
        return self.mobilenet(x)
    
# Load preprocessed train and validation data loaders
train_loader = torch.load('/kaggle/working/train_loader.pth')
val_loader = torch.load('/kaggle/working/val_loader.pth')

# Hyperparameters
num_classes = 3
learning_rate = 0.001
num_epochs = 50  # Increase the number of epochs if needed
patience = 10  # For early stopping

# Initialize the model, loss function, and optimizer
model = CustomModel(num_classes=num_classes)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# Learning rate scheduler
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)

# Move model to device (GPU if available)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# Variables to store metrics during training
train_losses = []
val_losses = []
train_accuracy = []
val_accuracy = []
best_val_acc = 0.0
early_stop_counter = 0

# Training loop with validation inside
for epoch in range(num_epochs):
    # Training step
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    print(f"Epoch {epoch+1}/{num_epochs} started.")  # Track epoch start
    for i, (inputs, labels) in enumerate(train_loader):
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad()  
        outputs = model(inputs)  # Forward pass
        loss = criterion(outputs, labels)  # Loss calculation
        loss.backward()  
        optimizer.step()  
        running_loss += loss.item()  # Accumulate loss
        _, predicted = torch.max(outputs, 1)  # Predictions
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

    # Store metrics for the epoch
    train_losses.append(running_loss / len(train_loader))
    train_accuracy.append(correct / total)

    # Validation step (moved inside the loop)
    model.eval()  # Set model to evaluation mode
    val_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():  # Disable gradient calculation for validation
        for inputs, labels in val_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            val_loss += loss.item()  # Accumulate validation loss
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    # Store validation metrics for the epoch
    val_losses.append(val_loss / len(val_loader))
    val_acc = correct / total
    val_accuracy.append(val_acc)

    # Print training and validation results for the entire epoch
    print(f'Epoch [{epoch+1}/{num_epochs}], '
          f'Train Loss: {train_losses[-1]:.4f}, Train Acc: {train_accuracy[-1]:.4f}, '
          f'Val Loss: {val_losses[-1]:.4f}, Val Acc: {val_acc:.4f}')

# Save the final model after training
torch.save(model.state_dict(), '/kaggle/working/final_model_weights_v4.pth')

# Save the final model after training
torch.save(model.state_dict(), '/kaggle/working/final_model_weights_v4.pth')

import matplotlib.pyplot as plt
# Plot accuracy and loss metrics
plt.figure(figsize=(15, 15))

# Plot Accuracy
plt.subplot(1, 2, 1)
plt.plot(train_accuracy, label='Train Accuracy', color='blue')
plt.plot(val_accuracy, label='Val Accuracy', color='orange')
plt.title('Accuracy per Epoch')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.savefig('/kaggle/working/training_metrics_accuracy.png')
plt.legend()

# Plot Loss
plt.subplot(1, 2, 2)
plt.plot(train_losses, label='Train Loss', color='blue')
plt.plot(val_losses, label='Val Loss', color='orange')
plt.title('Loss per Epoch')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()

plt.savefig('/kaggle/working/training_metrics.png')
plt.show()


