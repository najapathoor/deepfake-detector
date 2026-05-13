import torch
from torch.utils.data import DataLoader, Dataset
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from tqdm import tqdm
import os, shutil, cv2
from PIL import Image
from torchvision import transforms
from model.detector import DeepfakeDetector

# --- CONFIG ---
external_dataset_path = "test"
save_misclassified = True
misclassified_dir = "misclassified_samples"

# --- Load model ---
detector = DeepfakeDetector(weights="weights_finertuned.pth")
model = detector.model
model.eval()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
print(f"✅ Using device: {device}")

# --- Transform (same as detector.py) ---
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
])

# --- Video Dataset ---
class VideoFrameDataset(Dataset):
    def __init__(self, root, transform=None):
        self.samples   = []
        self.transform = transform
        self.classes   = sorted(os.listdir(root))  # ['fake', 'real']
        self.class_to_idx = {c: i for i, c in enumerate(self.classes)}
        for cls in self.classes:
            folder = os.path.join(root, cls)
            for f in os.listdir(folder):
                if f.lower().endswith(('.mp4', '.avi', '.mov', '.mkv')):
                    self.samples.append((os.path.join(folder, f), self.class_to_idx[cls]))

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

# --- Load dataset ---
external_dataset = VideoFrameDataset(root=external_dataset_path, transform=transform)
external_loader  = DataLoader(external_dataset, batch_size=32, shuffle=False)
class_names      = external_dataset.classes
print(f"Total videos: {len(external_dataset)} | Classes: {class_names}")

# --- Misclassified folder ---
if save_misclassified:
    if os.path.exists(misclassified_dir):
        shutil.rmtree(misclassified_dir)
    os.makedirs(misclassified_dir, exist_ok=True)

# --- Test loop ---
all_labels = []
all_preds  = []

with torch.no_grad():
    for imgs, labels in tqdm(external_loader, desc="Testing", unit="batch"):
        imgs, labels = imgs.to(device), labels.to(device)
        outputs = model(imgs)
        preds   = torch.argmax(outputs, dim=1)

        all_labels.extend(labels.cpu().numpy())
        all_preds.extend(preds.cpu().numpy())

        if save_misclassified:
            for i in range(len(preds)):
                if preds[i] != labels[i]:
                    orig_path = external_dataset.samples[len(all_labels) - len(preds) + i][0]
                    fname     = os.path.basename(orig_path)
                    pred_name = class_names[preds[i]]
                    true_name = class_names[labels[i]]
                    target_dir = os.path.join(misclassified_dir, f"pred_{pred_name}_true_{true_name}")
                    os.makedirs(target_dir, exist_ok=True)
                    shutil.copy(orig_path, os.path.join(target_dir, fname))

# --- Metrics ---
acc    = accuracy_score(all_labels, all_preds)
cm     = confusion_matrix(all_labels, all_preds)
report = classification_report(all_labels, all_preds, target_names=class_names)

print(f"\n✅ Accuracy: {acc*100:.2f}%")
print("\nConfusion Matrix:")
print(cm)
print("\nDetailed Report:")
print(report)

if save_misclassified:
    print(f"\n⚠️ Misclassified videos saved in: {misclassified_dir}/")