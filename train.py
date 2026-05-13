import torch
import torch.nn as nn
from torchvision import models, transforms, datasets
from torch.utils.data import DataLoader
from tqdm import tqdm  # <-- progress bar

# -----------------------
# Paths & Hyperparameters
# -----------------------
train_dir = "data/train"
val_dir = "data/val"
save_path = "weights.pth"

batch_size = 8
epochs = 3
lr = 1e-4
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# -----------------------
# Transforms & Datasets
# -----------------------
transform = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.ToTensor(),
    transforms.Normalize([0.5,0.5,0.5],[0.5,0.5,0.5])
])

train_dataset = datasets.ImageFolder(train_dir, transform=transform)
val_dataset = datasets.ImageFolder(val_dir, transform=transform)

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

# -----------------------
# Model
# -----------------------
model = models.resnet18(pretrained=True)
model.fc = nn.Linear(model.fc.in_features, 2)
model = model.to(device)

# -----------------------
# Loss & Optimizer
# -----------------------
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=lr)

# -----------------------
# Training loop with progress bar
# -----------------------
for epoch in range(epochs):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    print(f"\nEpoch {epoch+1}/{epochs}")
    for imgs, labels in tqdm(train_loader, desc="Training batches"):
        imgs, labels = imgs.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(imgs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        _, preds = torch.max(outputs,1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)

    train_acc = correct / total if total > 0 else 0
    avg_loss = running_loss / len(train_loader) if len(train_loader) > 0 else 0
    print(f"Epoch [{epoch+1}/{epochs}] Loss: {avg_loss:.4f} Accuracy: {train_acc:.4f}")

# -----------------------
# Save weights
# -----------------------
torch.save(model.state_dict(), save_path)
print(f"\n✅ Training complete. Weights saved to {save_path}")
