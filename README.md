<p align="center">

  <img src="https://img.shields.io/badge/Waste%20Classification-4CAF50?style=for-the-badge&logo=recycle&logoColor=white&colorA=4CAF50&colorB=81C784" />

  <img src="https://img.shields.io/badge/Computer%20Vision-2196F3?style=for-the-badge&logo=opencv&logoColor=white&colorA=2196F3&colorB=64B5F6" />

  <img src="https://img.shields.io/badge/Machine%20Learning-FF9800?style=for-the-badge&logo=tensorflow&logoColor=white&colorA=FF9800&colorB=FFB74D" />

  <img src="https://img.shields.io/badge/Deep%20Learning-9C27B0?style=for-the-badge&logo=keras&logoColor=white&colorA=9C27B0&colorB=BA68C8" />

  <img src="https://img.shields.io/badge/TFLite-0288D1?style=for-the-badge&logo=google&logoColor=white&colorA=0288D1&colorB=4FC3F7" />

  <img src="https://img.shields.io/badge/VLM-673AB7?style=for-the-badge&logo=huggingface&logoColor=white&colorA=673AB7&colorB=9575CD" />

  <img src="https://img.shields.io/badge/SmolVLM-512DA8?style=for-the-badge&logo=huggingface&logoColor=white&colorA=512DA8&colorB=9575CD" />

  <img src="https://img.shields.io/badge/Hugging%20Face-F9D649?style=for-the-badge&logo=huggingface&logoColor=000&colorA=F9D649&colorB=FDD835" />

  <img src="https://img.shields.io/badge/Real--Time-00C853?style=for-the-badge&logo=fastapi&logoColor=white&colorA=00C853&colorB=69F0AE" />

  <img src="https://img.shields.io/badge/IoT-455A64?style=for-the-badge&logo=raspberrypi&logoColor=white&colorA=455A64&colorB=90A4AE" />

</p>





## ♻️ Waste Segregation Bot: Dual Classification System

This repository presents a powerful, dual-approach solution for real-time waste classification into five core categories: Organic, Inorganic, Metal, Electronic, and Others.

The project is designed for both high-performance deployment on constrained devices and robust, flexible classification using modern AI models.

## 🌟 Architecture Overview

#### This project features two independent and complementary pipelines:

1. Classical Computer Vision (TensorFlow/Keras)

##### Core Feature: High speed, low memory footprint, ideal for edge computing.

2. Modern Vision-Language Model (Hugging Face SmolVLM)

##### Core Feature: Exceptional generalization and language-guided reasoning.

## 📦 Core Waste Categories

#### Organic

Biodegradable waste.

Food scraps, leaves, paper towels, wood.

#### Inorganic

Non-biodegradable waste.

Plastic bottles, glass, styrofoam, cardboard.

#### Others

Miscellaneous waste not covered above.

Textiles, rubber, mixed materials.

## 🚀 Setup and Installation

#### Prerequisites

You need a Python environment (3.8+ recommended).

#### 1. Update Dependencies

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



#### 2. Dataset Setup (For Training Only)

For the classical model training (main.py), you must provide your dataset.

Create a directory named RealWaste (or any name, and update DATA_DIR in main.py).

Place your image data inside, organized into subdirectories representing your original class names (e.g., RealWaste/Plastic, RealWaste/Food Organics).

## 🛠️ Pipeline 1: Classical ML (ResNet50 + TFLite)

This pipeline is ideal for training a highly optimized model for embedded systems.

#### A. Training the Model

The main.py script handles data loading, class mapping, transfer learning with ResNet50, and TFLite conversion.

Run Training:

python "waste classification/src/main.py"



Output: This will generate two files:

waste_classifier.h5 (Full Keras Model)

waste_classifier.tflite (Optimized TFLite Model)

#### B. Real-Time Inference with Keras/TFLite

The inference.py script uses the trained model to perform real-time classification from your webcam.

Set Model Path: Ensure MODEL_PATH in inference.py points to your saved model (e.g., "waste_classifier.tflite").

Run Inference:

python "waste classification/src/inference.py"



## 🧠 Pipeline 2: Modern VLM (SmolVLM-256M-Instruct)

This pipeline uses a powerful Vision-Language Model for potentially higher out-of-distribution performance without needing a local dataset.

#### A. Single-Shot Classification (inference2.py)

This script captures a single image from the webcam and classifies it.

Run Script:

python "waste classification/src/inference2.py"



Output: Prints the predicted category to the console.

#### B. Stable Real-Time Classification (test.py)

This script provides a continuous, more stable real-time webcam feed with classification, using a prediction history (deque) to stabilize the output and reduce flicker.

Run Script:

python "waste classification/src/test.py"



Usage: A window will open showing the webcam feed with a blue bounding box (Region of Interest) and the stabilized prediction text. Press q to quit.

## 📁 Repository Structure

```bash
.
└── waste-classification/
    ├── requirements.txt
    └── src/
        ├── main.py                     # Entry point for running the full pipeline
        ├── inference.py                # Image classification using Keras/TFLite model
        ├── inference2.py               # Vision-Language Model (SmolVLM) single-frame inference
        └── test.py                     # Real-time webcam classification using SmolVLM
```


## 📄 License

This project is licensed under the **MIT License**

See the [`LICENSE`](./LICENSE) file for full details.


