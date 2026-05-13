import torch
import torch.nn as nn
from torchvision import models, transforms
from torch.utils.data import DataLoader, Dataset
import numpy as np
from sklearn.metrics import roc_curve, accuracy_score, classification_report
from PIL import Image
import cv2
import os

# ─────────────────────────────────────────────
# EDIT THESE TWO PATHS BEFORE RUNNING
# ─────────────────────────────────────────────
MODEL_PATH = "weights_finertuned.pth"
TEST_DIR   = "test"   # expects test_subset/fake/ and test_subset/real/
# ─────────────────────────────────────────────

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

class VideoFrameDataset(Dataset):
    def __init__(self, root, transform=None):
        self.samples   = []   # (path, label)
        self.transform = transform
        # fake=0, real=1  — same as ImageFolder default alphabetical order
        for label, cls in enumerate(['fake', 'real']):
            folder = os.path.join(root, cls)
            for f in os.listdir(folder):
                if f.lower().endswith(('.mp4', '.avi', '.mov', '.mkv')):
                    self.samples.append((os.path.join(folder, f), label))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        cap = cv2.VideoCapture(path)
        ret, frame = cap.read()
        cap.release()
        if not ret:
            raise RuntimeError(f"Could not read: {path}")
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image = Image.fromarray(frame)
        if self.transform:
            image = self.transform(image)
        return image, label

# Load model
model = models.resnet18(pretrained=False)
model.fc = nn.Linear(model.fc.in_features, 2)
model.load_state_dict(torch.load(MODEL_PATH, map_location=torch.device('cpu')))
model.eval()

# Load dataset
test_dataset = VideoFrameDataset(root=TEST_DIR, transform=transform)
test_loader  = DataLoader(test_dataset, batch_size=16, shuffle=False)

print(f"Total videos found: {len(test_dataset)}")
print("Class mapping: fake=0, real=1")

all_labels = []
all_probs  = []

with torch.no_grad():
    for images, labels in test_loader:
        outputs    = model(images)
        probs      = torch.softmax(outputs, dim=1)
        fake_probs = probs[:, 0].numpy()
        all_probs.extend(fake_probs)
        all_labels.extend(labels.numpy())

all_labels = np.array(all_labels)
all_probs  = np.array(all_probs)

binary_labels = (all_labels == 0).astype(int)  # 1=fake, 0=real

# ── Accuracy at default threshold ────────────
preds_default = (all_probs >= 0.5).astype(int)
acc_default   = accuracy_score(binary_labels, preds_default)
print(f"\nAccuracy at default threshold (0.5): {acc_default*100:.2f}%")

# ── Best threshold ────────────────────────────
fpr, tpr, thresholds = roc_curve(binary_labels, all_probs)
best_thresh, best_acc = 0.5, 0.0
for t in thresholds:
    preds = (all_probs >= t).astype(int)
    acc   = accuracy_score(binary_labels, preds)
    if acc > best_acc:
        best_acc, best_thresh = acc, t

print(f"Best threshold found:                 {best_thresh:.4f}")
print(f"Best accuracy at that threshold:      {best_acc*100:.2f}%")

# ── Classification report ─────────────────────
print("\n── Classification Report at Best Threshold ──")
best_preds = (all_probs >= best_thresh).astype(int)
print(classification_report(binary_labels, best_preds, target_names=['Real', 'Fake']))

# ── Threshold scan ────────────────────────────
print("── Accuracy across thresholds ──")
print(f"{'Threshold':>10} {'Accuracy':>10}")
for t in np.arange(0.3, 0.8, 0.05):
    preds  = (all_probs >= t).astype(int)
    acc    = accuracy_score(binary_labels, preds)
    marker = " <-- default" if abs(t - 0.5) < 0.01 else ""
    print(f"{t:>10.2f} {acc*100:>9.2f}%{marker}")