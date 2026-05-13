import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image

class DeepfakeDetector:
    def __init__(self, weights="weights_finertuned.pth"):
        # Load ResNet18 pretrained
        self.model = models.resnet18(pretrained=True)
        self.model.fc = nn.Linear(self.model.fc.in_features, 2)
        self.model.load_state_dict(torch.load(weights, map_location="cpu"))
        self.model.eval()

        # Image transform
        self.transform = transforms.Compose([
            transforms.Resize((224,224)),
            transforms.ToTensor(),
            transforms.Normalize([0.5,0.5,0.5],[0.5,0.5,0.5])
        ])

    def predict(self, image_path, return_confidence=False):
        # Make sure this whole block is indented under the function
        image = Image.open(image_path).convert("RGB")
        x = self.transform(image).unsqueeze(0)
        with torch.no_grad():
            preds = self.model(x)
            probs = torch.softmax(preds, dim=1)
            label = torch.argmax(probs, 1).item()
        
        if return_confidence:
            confidence = probs[0][label].item()
            return ("REAL" if label == 1 else "FAKE"), confidence
        
        return "REAL" if label == 1 else "FAKE"
