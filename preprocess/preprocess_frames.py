# save as C:\deepcheck\preprocess\preprocess_frames_save.py
from PIL import Image
import os
from pathlib import Path
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor

SRC_ROOT = r"C:\deepcheck\frames"            # original extracted frames
DST_ROOT = r"C:\deepcheck\frames_preproc"    # preprocessed frames
TARGET_SIZE = 224
RESIZE_SHORTER = 256   # resize shorter side to this, then center-crop to TARGET_SIZE
JPEG_QUALITY = 85

def process_image(args):
    src_path, dst_path = args
    try:
        dst_path.parent.mkdir(parents=True, exist_ok=True)
        with Image.open(src_path) as im:
            im = im.convert("RGB")
            w, h = im.size
            # Resize preserving aspect ratio: shorter side -> RESIZE_SHORTER
            if w < h:
                new_w = RESIZE_SHORTER
                new_h = int(h * RESIZE_SHORTER / w)
            else:
                new_h = RESIZE_SHORTER
                new_w = int(w * RESIZE_SHORTER / h)
            im = im.resize((new_w, new_h), Image.BILINEAR)
            # Center crop TARGET_SIZE x TARGET_SIZE
            left = (new_w - TARGET_SIZE) // 2
            top = (new_h - TARGET_SIZE) // 2
            im = im.crop((left, top, left + TARGET_SIZE, top + TARGET_SIZE))
            im.save(dst_path, format="JPEG", quality=JPEG_QUALITY, optimize=True)
        return True
    except Exception as e:
        return False

def gather_all_images(src_root, dst_root):
    tasks = []
    for root, _, files in os.walk(src_root):
        for f in files:
            if f.lower().endswith((".jpg", ".jpeg", ".png")):
                src = Path(root) / f
                rel = src.relative_to(src_root)
                dst = Path(dst_root) / rel
                tasks.append((src, dst))
    return tasks

if __name__ == "__main__":
    tasks = gather_all_images(SRC_ROOT, DST_ROOT)
    print(f"Found {len(tasks)} images to preprocess.")
    # Use ProcessPoolExecutor for CPU-bound resizing (adjust workers to your CPU)
    workers = max(1, os.cpu_count() - 1)
    with ProcessPoolExecutor(max_workers=workers) as ex:
        for ok in tqdm(ex.map(process_image, tasks), total=len(tasks), desc="Preprocessing"):
            pass

    print("Done. Preprocessed frames saved to:", DST_ROOT)
