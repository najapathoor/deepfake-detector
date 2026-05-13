import os
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, models, transforms
from torch.utils.data import DataLoader, SubsetRandomSampler
from tqdm import tqdm
import numpy as np
from sklearn.metrics import accuracy_score

# ----------------------------
# Configuration
# ----------------------------
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
TRAIN_DIR = "dataset_new/train"
VAL_DIR = "dataset_new/val"
BATCH_SIZE = 32
EPOCHS = 10
LEARNING_RATE = 1e-4
SAMPLES_PER_EPOCH = 2000  # number of random images each epoch
OLD_WEIGHTS_PATH = "weights_finetuned.pth"   # existing model
NEW_WEIGHTS_PATH = "weights_finertuned.pth"  # new fine-tuned model will be saved here

# ----------------------------
# Data Transforms
# ----------------------------
train_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(10),
    transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3),
    transforms.ToTensor(),
    transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
])

val_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
])

# ----------------------------
# Datasets
# ----------------------------
train_dataset = datasets.ImageFolder(TRAIN_DIR, transform=train_transform)
val_dataset = datasets.ImageFolder(VAL_DIR, transform=val_transform)

train_size = len(train_dataset)
val_size = len(val_dataset)
print(f"🟩 Total training images available: {train_size}")
print(f"🟩 Total validation images available: {val_size}")

# Function to select a random subset of images each epoch
def get_random_sampler():
    indices = np.random.choice(train_size, SAMPLES_PER_EPOCH, replace=False)
    return SubsetRandomSampler(indices)

# ----------------------------
# Model Setup
# ----------------------------
model = models.resnet18(pretrained=False)
num_features = model.fc.in_features
model.fc = nn.Linear(num_features, 2)

# ✅ Load your existing fine-tuned weights
if os.path.exists(OLD_WEIGHTS_PATH):
    print("🔁 Loading existing fine-tuned weights...")
    model.load_state_dict(torch.load(OLD_WEIGHTS_PATH, map_location=DEVICE))
else:
    print("⚠️ No previous weights found. Training from scratch...")

model = model.to(DEVICE)

# ----------------------------
# Loss & Optimizer
# ----------------------------
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

# ----------------------------
# Training Loop
# ----------------------------
best_val_acc = 0.0

for epoch in range(EPOCHS):
    print(f"\n🟢 Epoch {epoch+1}/{EPOCHS}")

    # Random new subset each epoch
    train_sampler = get_random_sampler()
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, sampler=train_sampler)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

    # ---- Training ----
    model.train()
    train_loss = 0.0
    correct = 0
    total = 0

    for images, labels in tqdm(train_loader, desc="Training", leave=False):
        images, labels = images.to(DEVICE), labels.to(DEVICE)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        _, preds = torch.max(outputs, 1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)

    train_acc = correct / total
    print(f"🧠 Training Loss: {train_loss/len(train_loader):.4f} | Accuracy: {train_acc*100:.2f}%")

    # ---- Validation ----
    model.eval()
    val_correct = 0
    val_total = 0
    preds_all = []
    labels_all = []

    with torch.no_grad():
        for images, labels in tqdm(val_loader, desc="Validating", leave=False):
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            outputs = model(images)
            _, preds = torch.max(outputs, 1)
            val_correct += (preds == labels).sum().item()
            val_total += labels.size(0)
            preds_all.extend(preds.cpu().numpy())
            labels_all.extend(labels.cpu().numpy())

    val_acc = val_correct / val_total
    print(f"📊 Validation Accuracy: {val_acc*100:.2f}%")

    # ✅ Save only best model (with new name)
    if val_acc > best_val_acc:
        best_val_acc = val_acc
        torch.save(model.state_dict(), NEW_WEIGHTS_PATH)
        print(f"✅ New fine-tuned model saved as {NEW_WEIGHTS_PATH} with val_acc: {best_val_acc*100:.2f}%")

print("\n🎯 Training complete!")
print(f"🏆 Best Validation Accuracy: {best_val_acc*100:.2f}%")
print(f"📁 Model saved to: {NEW_WEIGHTS_PATH}")
