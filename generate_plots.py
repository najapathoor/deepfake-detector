import torch
import torch.nn as nn
from torchvision import models, transforms
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
from sklearn.metrics import roc_curve, auc
import os

# ─────────────────────────────────────────────
# EDIT THESE TWO PATHS BEFORE RUNNING
# ─────────────────────────────────────────────
MODEL_PATH = "weights_finertuned.pth"   # path to your saved model
TEST_DIR   = "test_subset"            # folder with subfolders: real/ and fake/
# ─────────────────────────────────────────────

# ── 1. Training history (your epoch data) ────
epochs      = list(range(1, 11))
train_acc   = [98.90, 98.85, 98.85, 98.70, 98.75, 99.15, 98.75, 99.00, 98.85, 98.90]
val_acc     = [75.57, 77.29, 77.95, 76.34, 77.44, 75.71, 77.07, 76.10, 78.31, 77.83]
train_loss  = [0.0279, 0.0209, 0.0273, 0.0248, 0.0274, 0.0187, 0.0323, 0.0240, 0.0233, 0.0261]

# ── 2. Plot training curves ───────────────────
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
fig.suptitle("DeepCheck — Training Dynamics", fontsize=14, fontweight='bold', y=1.01)

# Accuracy subplot
ax1.plot(epochs, train_acc, 'o-', color='#2E86AB', linewidth=2, markersize=6, label='Training Accuracy')
ax1.plot(epochs, val_acc,   's--', color='#E84855', linewidth=2, markersize=6, label='Validation Accuracy')
ax1.axvline(x=9, color='gray', linestyle=':', linewidth=1.5, label='Best checkpoint (Epoch 9)')
ax1.annotate('78.31%', xy=(9, 78.31), xytext=(7.2, 79.2),
             fontsize=9, color='#E84855',
             arrowprops=dict(arrowstyle='->', color='#E84855', lw=1.2))
ax1.set_xlabel('Epoch', fontsize=12)
ax1.set_ylabel('Accuracy (%)', fontsize=12)
ax1.set_title('Training vs. Validation Accuracy', fontsize=12)
ax1.set_xticks(epochs)
ax1.set_ylim(70, 102)
ax1.legend(fontsize=10)
ax1.grid(True, alpha=0.3)

# Loss subplot
ax2.plot(epochs, train_loss, 'o-', color='#F18F01', linewidth=2, markersize=6, label='Training Loss')
ax2.axvline(x=9, color='gray', linestyle=':', linewidth=1.5, label='Best checkpoint (Epoch 9)')
ax2.set_xlabel('Epoch', fontsize=12)
ax2.set_ylabel('Loss', fontsize=12)
ax2.set_title('Training Loss', fontsize=12)
ax2.set_xticks(epochs)
ax2.legend(fontsize=10)
ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('training_curves.png', dpi=300, bbox_inches='tight')
plt.close()
print("Saved: training_curves.png")

# ── 3. ROC Curve ─────────────────────────────
# Load model
model = models.resnet18(pretrained=False)
model.fc = nn.Linear(model.fc.in_features, 2)
model.load_state_dict(torch.load(MODEL_PATH, map_location=torch.device('cpu')))
model.eval()

# Load test data
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

test_dataset = ImageFolder(root=TEST_DIR, transform=transform)
test_loader  = DataLoader(test_dataset, batch_size=16, shuffle=False)

# Check class mapping — fake should be index 0, real index 1
# ImageFolder sorts alphabetically: fake=0, real=1
print("Class mapping:", test_dataset.class_to_idx)

all_labels = []
all_probs  = []

with torch.no_grad():
    for images, labels in test_loader:
        outputs = model(images)
        probs   = torch.softmax(outputs, dim=1)
        # probability of being FAKE (class index 0)
        fake_probs = probs[:, 0].numpy()
        all_probs.extend(fake_probs)
        # convert labels: fake=0 stays 0 (positive class), real=1 stays 1
        # for ROC we want: 1 = fake (positive), 0 = real (negative)
        binary_labels = (labels.numpy() == 0).astype(int)
        all_labels.extend(binary_labels)

all_labels = np.array(all_labels)
all_probs  = np.array(all_probs)

fpr, tpr, thresholds = roc_curve(all_labels, all_probs)
roc_auc = auc(fpr, tpr)

# Plot ROC
fig, ax = plt.subplots(figsize=(7, 6))
ax.plot(fpr, tpr, color='#2E86AB', lw=2.5,
        label=f'ROC Curve (AUC = {roc_auc:.4f})')
ax.plot([0, 1], [0, 1], color='gray', lw=1.5, linestyle='--', label='Random Classifier')
ax.fill_between(fpr, tpr, alpha=0.08, color='#2E86AB')
ax.set_xlabel('False Positive Rate', fontsize=13)
ax.set_ylabel('True Positive Rate', fontsize=13)
ax.set_title('ROC Curve — DeepCheck (ResNet18)', fontsize=13, fontweight='bold')
ax.legend(loc='lower right', fontsize=11)
ax.grid(True, alpha=0.3)
ax.set_xlim([0.0, 1.0])
ax.set_ylim([0.0, 1.02])

plt.tight_layout()
plt.savefig('roc_curve.png', dpi=300, bbox_inches='tight')
plt.close()
print(f"Saved: roc_curve.png  |  AUC = {roc_auc:.4f}")
print("\nDone. Add training_curves.png and roc_curve.png to your LaTeX folder.")
