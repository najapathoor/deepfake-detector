import numpy as np
import tensorflow as tf
from tensorflow.keras.applications.xception import Xception, preprocess_input
from tensorflow.keras.preprocessing import image
from pathlib import Path
import os

def extract_features(face_dir, output_npy):
    model = Xception(weights="imagenet", include_top=False, pooling="avg")
    features = []

    for face_file in Path(face_dir).glob("*.jpg"):
        img = image.load_img(face_file, target_size=(299, 299))
        x = image.img_to_array(img)
        x = np.expand_dims(x, axis=0)
        x = preprocess_input(x)
        feat = model.predict(x, verbose=0)
        features.append(feat.squeeze())

    features = np.array(features)
    np.save(output_npy, features)
    print(f"Saved {features.shape} features to {output_npy}")

if __name__ == "__main__":
    face_dir = r"C:\deepcheck\faces\sample"
    out_file = r"C:\deepcheck\features\sample.npy"
    extract_features(face_dir, out_file)
