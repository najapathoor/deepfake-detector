# Deepfake Video Detection Using AI

An AI-powered deepfake video detection system built using ResNet18 and PyTorch.  
This project detects manipulated/fake videos by analyzing extracted video frames using deep learning techniques.

---

## Features

- Deepfake video classification
- Frame extraction and preprocessing
- ResNet18-based CNN model
- Fine-tuning support
- Accuracy and performance visualization
- Confidence score prediction
- Graph generation for analysis

---

## Tech Stack

- Python
- PyTorch
- OpenCV
- NumPy
- Matplotlib
- Streamlit

---

## Project Structure

```text
deepfake-detector/
│
├── preprocess/
├── model/
├── graphs/
├── screenshots/
├── train.py
├── fine_tune.py
├── accuracy.py
├── generate_plots.py
├── app.py
└── README.md
```

---

## Model Architecture

- Base Model: ResNet18
- Loss Function: CrossEntropyLoss
- Optimizer: Adam
- Framework: PyTorch

---

## Workflow

1. Video input
2. Frame extraction
3. Preprocessing
4. Feature extraction using ResNet18
5. Classification into:
   - Real
   - Fake
6. Confidence score generation

---

## Results

The model was trained and evaluated on deepfake datasets and achieved promising classification performance.

Performance graphs and evaluation outputs are available in the `graphs/` folder.

---

## Screenshots

### Application Interface

Add screenshots from the `screenshots/` folder here.

Example:

```md
![App Screenshot](screenshots/your_image.png)
```

---

## Future Improvements

- Real-time webcam detection
- Attention mechanisms
- Transformer-based architectures
- Mobile deployment
- Improved dataset balancing
- GPU based training to increase efficiency

---

## Installation

Clone the repository:

```bash
git clone https://github.com/najapathoor/deepfake-detector.git
```

Install dependencies:

```bash
pip install -r requirements.txt
```

Run the application:

```bash
streamlit run app.py
```

---

## Author

Fathima Naja  
Computer Science and Business Systems Student