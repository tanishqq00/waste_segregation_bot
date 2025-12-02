# ♻️ Waste Segregation Bot: Dual Classification System

This repository presents a dual-approach solution for real-time waste classification into five core categories: Organic, Inorganic, Metal, Electronic, and Others.

It provides two distinct pipelines:

## Classical Computer Vision (TensorFlow/Keras): Optimized for training custom models and deployment on edge device

ces (via TFLite).

Modern Vision-Language Model (Hugging Face SmolVLM): Leveraging zero-shot natural language prompts for robust, high-accuracy classification without extensive custom training.

# 📦 Core Waste Categories

### Organic

Biodegradable waste.

Food scraps, leaves, paper towels, wood.

### Inorganic

Non-biodegradable waste.

Plastic bottles, glass, styrofoam, cardboard.

### Others

Miscellaneous waste not covered above.

Textiles, rubber, mixed materials.

# 🚀 Setup and Installation

### Prerequisites

You need a Python environment (3.8+ recommended).

### 1. Update Dependencies

Since this project uses both TensorFlow/Keras and PyTorch/Hugging Face, you need to install all required libraries.

Note: The provided requirements.txt is incomplete. Please use the expanded set below.

Create a new file named full_requirements.txt and add the following:

tensorflow
tensorflow-hub
tensorflow-model-optimization
numpy
matplotlib
scikit-learn
pillow
tqdm
opencv-python
torch
transformers



Now, install them:

pip install -r full_requirements.txt



### 2. Dataset Setup (For Training Only)

For the classical model training (main.py), you must provide your dataset.

Create a directory named RealWaste (or any name, and update DATA_DIR in main.py).

Place your image data inside, organized into subdirectories representing your original class names (e.g., RealWaste/Plastic, RealWaste/Food Organics).

# 🛠️ Pipeline 1: Classical ML (ResNet50 + TFLite)

This pipeline is ideal for training a highly optimized model for embedded systems.

### A. Training the Model

The main.py script handles data loading, class mapping, transfer learning with ResNet50, and TFLite conversion.

Run Training:

python "waste classification/src/main.py"



Output: This will generate two files:

waste_classifier.h5 (Full Keras Model)

waste_classifier.tflite (Optimized TFLite Model)

### B. Real-Time Inference with Keras/TFLite

The inference.py script uses the trained model to perform real-time classification from your webcam.

Set Model Path: Ensure MODEL_PATH in inference.py points to your saved model (e.g., "waste_classifier.tflite").

Run Inference:

python "waste classification/src/inference.py"



# 🧠 Pipeline 2: Modern VLM (SmolVLM-256M-Instruct)

This pipeline uses a powerful Vision-Language Model for potentially higher out-of-distribution performance without needing a local dataset.

### A. Single-Shot Classification (inference2.py)

This script captures a single image from the webcam and classifies it.

Run Script:

python "waste classification/src/inference2.py"



Output: Prints the predicted category to the console.

### B. Stable Real-Time Classification (test.py)

This script provides a continuous, more stable real-time webcam feed with classification, using a prediction history (deque) to stabilize the output and reduce flicker.

Run Script:

python "waste classification/src/test.py"



Usage: A window will open showing the webcam feed with a blue bounding box (Region of Interest) and the stabilized prediction text. Press q to quit.

# 📁 Repository Structure

.

└── waste classification/
    ├── requirements.txt
    └── src/
        ├── main.py            
        ├── inference.py        # using Keras or TFLite
        ├── inference2.py       # VLM: Single-frame classification using SmolVLM
        └── test.py             # VLM: Stable, real-time webcam classification using SmolVLM

