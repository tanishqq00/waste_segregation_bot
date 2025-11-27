♻️ Autonomous Waste Segregation Bot
Deep Learning–Powered Biodegradable vs Non-Biodegradable Waste Classifier

(TensorFlow + ResNet50 + TFLite)

📌 Overview

The Autonomous Waste Segregation Bot uses a deep-learning image classifier to automatically categorize waste into:

Biodegradable

Non-Biodegradable

This model is built using TensorFlow and ResNet50 (transfer learning), and can be deployed on edge devices using TensorFlow Lite (.tflite) for real-time waste detection.

The project is suitable for:

Smart bin automation

IoT waste sorting

Robotics applications

Mobile/embedded ML deployments

🧠 Model Features

✔ Transfer learning using ResNet50
✔ Custom dataset loading with tf.data
✔ Data augmentation (flip, rotation, zoom)
✔ Automatic mapping from raw dataset classes → 2 target classes
✔ Fine-tuning support for high accuracy
✔ Conversion to TensorFlow Lite (quantized)
✔ Modular class-based design for reusability

📂 Project Structure
Autonomous-Waste-Seg-Bot/
│── main.py
│── waste_classifier.py   # (your class file)
│── README.md
│── RealWaste/            # dataset folder
│   ├── biodegradable/
│   ├── non_biodegradable/
│── models/
│   ├── waste_classifier.h5
│   ├── waste_classifier.tflite
│── .gitignore

📦 Dependencies

Install the required libraries:

pip install tensorflow
pip install numpy
pip install pillow


TensorFlow GPU is optional but recommended.

▶️ How to Run
1. Place your dataset

Your dataset directory should look like:

RealWaste/
 ├── biodegradable/
 ├── non_biodegradable/

2. Run the training script
python main.py

3. Output Files

After training, the following files are generated:

waste_classifier.h5 — full TensorFlow model

waste_classifier.tflite — optimized for mobile/edge devices

🏗 Architecture Overview

Base Model: ResNet50 (ImageNet pre-trained)

Image Size: 224×224

Batch Size: 32

Loss: categorical crossentropy

Optimizer: Adam

Final Layer: Dense → Softmax (2 classes)

🚀 Model Deployment

Use the exported TFLite model for:

✔ Arduino Nano 33 BLE Sense
✔ Raspberry Pi + Coral TPU
✔ Android App (TensorFlow Lite)
✔ Jetson Nano

To load the TFLite model:

import tensorflow as tf

interpreter = tf.lite.Interpreter(model_path="waste_classifier.tflite")
interpreter.allocate_tensors()

📊 Training Process

Phase 1: Train classifier head (ResNet frozen)

Phase 2: Fine-tune deeper layers for higher accuracy

Early stopping and learning rate scheduling are used

Dataset is automatically mapped into 2 classes

🖼 Sample Predictions (example)
result = classifier.predict("sample.jpg")
print("Predicted class:", result)


Output:

Biodegradable

📜 License

This project is released under the MIT License.
You may use, modify, and distribute it freely with attribution.

🤝 Contributing

Pull requests are welcome!
Feel free to open an issue for new feature suggestions or bugs.

💬 Contact

If you need help with deployment, TFLite conversion, or dataset preparation, feel free to ask!