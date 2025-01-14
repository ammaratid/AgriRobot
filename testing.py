!pip install torchviz torchsummary

import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, roc_curve, auc
from torchsummary import summary
from torchvision import models  # Add this to use MobileNetV2

# Load the custom model
class CustomModel(nn.Module):
    def __init__(self, num_classes=3):
        super(CustomModel, self).__init__()
        self.mobilenet = models.mobilenet_v2(pretrained=True)
        self.mobilenet.classifier[1] = nn.Linear(self.mobilenet.last_channel, num_classes)

    def forward(self, x):
        return self.mobilenet(x)
    
# Load preprocessed test data loader
test_loader = torch.load('/kaggle/working/test_loader.pth')

# Load the trained model
model = CustomModel(num_classes=3)
model.load_state_dict(torch.load('/kaggle/working/final_model_weights_v4.pth'))
model.eval()  # Set the model to evaluation mode

# Set device to GPU if available, otherwise CPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)  # Make sure model is on the correct device

# Summarize the model (Ensure the input size matches your dataset's image size)
summary(model, input_size=(3, 256, 256))  # Adjust the input size if necessary

# Variables for storing true labels and predictions
all_labels = []
all_preds = []

# Evaluate model on test set
model.eval()  # Set model to evaluation mode
with torch.no_grad():  # Disable gradient calculations for evaluation
    for inputs, labels in test_loader:
        inputs, labels = inputs.to(device), labels.to(device)  # Move inputs and labels to the correct device
        
        outputs = model(inputs)
        _, predicted = torch.max(outputs, 1)  # Get the index of the max log-probability
        
        # Move predictions and labels back to CPU to convert to numpy
        all_labels.extend(labels.cpu().numpy())
        all_preds.extend(predicted.cpu().numpy())

# Convert lists to numpy arrays
all_labels = np.array(all_labels)
all_preds = np.array(all_preds)

# Calculate metrics
accuracy = accuracy_score(all_labels, all_preds)
precision = precision_score(all_labels, all_preds, average='weighted')
recall = recall_score(all_labels, all_preds, average='weighted')
f1 = f1_score(all_labels, all_preds, average='weighted')

print(f"Accuracy: {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall: {recall:.4f}")
print(f"F1-Score: {f1:.4f}")

# Confusion Matrix
cm = confusion_matrix(all_labels, all_preds)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.title('Confusion Matrix')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.savefig('/kaggle/working/confusion_matrix.png')
plt.show()

from sklearn.preprocessing import label_binarize

# Binarize the labels for multi-class ROC curve
all_labels_bin = label_binarize(all_labels, classes=[0, 1, 2])  # Adjust the classes as per your case

# Compute ROC curve and ROC area for each class
n_classes = all_labels_bin.shape[1]
fpr = dict()
tpr = dict()
roc_auc = dict()

for i in range(n_classes):
    fpr[i], tpr[i], _ = roc_curve(all_labels_bin[:, i], all_preds == i)
    roc_auc[i] = auc(fpr[i], tpr[i])

# Plot ROC curve for each class
plt.figure(figsize=(10, 8))
for i in range(n_classes):
    plt.plot(fpr[i], tpr[i], label=f'Class {i} ROC curve (area = {roc_auc[i]:0.2f})')

plt.plot([0, 1], [0, 1], color='red', linestyle='--')
plt.title('Receiver Operating Characteristic (ROC) - Multiclass')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.legend(loc='lower right')
plt.savefig('/kaggle/working/roc_curve_multiclass.png')

