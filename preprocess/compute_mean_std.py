import os
import random
from PIL import Image
import torch
from torchvision import transforms
from tqdm import tqdm

# Path to your preprocessed frames folder
ROOT = r"C:\deepcheck\frames_preproc"

# Transform to convert image to tensor [C,H,W] with values in [0,1]
transform = transforms.ToTensor()

sum_ = torch.zeros(3)
sumsq = torch.zeros(3)
num_pixels = 0

# Collect all image paths
all_images = []
for root, _, files in os.walk(ROOT):
    for f in files:
        if f.lower().endswith((".jpg", ".jpeg", ".png")):
            all_images.append(os.path.join(root, f))

print(f"Found {len(all_images)} images.")

# Randomly sample up to 5000 images (or fewer if dataset smaller)
sample_size = min(5000, len(all_images))
sample_images = random.sample(all_images, sample_size)
print(f"Sampling {sample_size} images to estimate mean & std...")

# Loop with progress bar
for img_path in tqdm(sample_images, desc="Processing images"):
    img = Image.open(img_path).convert("RGB")
    t = transform(img)  # tensor [C,H,W] values 0–1
    c, h, w = t.shape
    pixels = h * w
    sum_ += t.view(3, -1).sum(dim=1)
    sumsq += (t.view(3, -1) ** 2).sum(dim=1)
    num_pixels += pixels

# Compute mean and std
mean = sum_ / num_pixels
std = (sumsq / num_pixels - mean ** 2).sqrt()

print("\n✅ Estimated dataset statistics (using sample):")
print("Mean:", mean.tolist())
print("Std :", std.tolist())
