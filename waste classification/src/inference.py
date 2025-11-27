import cv2
import numpy as np
import tensorflow as tf
from tensorflow import keras
import time
from pathlib import Path

class WasteInference:
    def __init__(self, model_path, img_size=(224, 224), use_tflite=True):
        self.img_size = img_size
        self.use_tflite = use_tflite
        self.class_names = ['Organic', 'Inorganic', 'Metal', 'Electronic', 'Others']
        
        # Load model
        if use_tflite and model_path.endswith('.tflite'):
            self.interpreter = tf.lite.Interpreter(model_path=model_path)
            self.interpreter.allocate_tensors()
            self.input_details = self.interpreter.get_input_details()
            self.output_details = self.interpreter.get_output_details()
            self.model = None
        else:
            self.model = keras.models.load_model(model_path)
            self.interpreter = None
        
        # For FPS calculation
        self.fps_counter = 0
        self.start_time = time.time()
        
    def preprocess_frame(self, frame):
        """Preprocess camera frame for inference"""
        # Resize frame
        resized = cv2.resize(frame, self.img_size)
        
        # Convert BGR to RGB
        rgb_frame = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
        
        # Normalize to [0, 1] and expand dimensions
        processed = np.expand_dims(rgb_frame.astype(np.float32) / 255.0, axis=0)
        
        # Apply ResNet50 preprocessing
        processed = keras.applications.resnet50.preprocess_input(processed * 255.0)
        
        return processed
    
    def predict(self, processed_frame):
        """Make prediction using loaded model"""
        if self.use_tflite and self.interpreter:
            # TFLite inference
            self.interpreter.set_tensor(self.input_details[0]['index'], processed_frame)
            self.interpreter.invoke()
            predictions = self.interpreter.get_tensor(self.output_details[0]['index'])
        else:
            # Keras model inference
            predictions = self.model.predict(processed_frame, verbose=0)
        
        return predictions[0]
    
    def get_prediction_text(self, predictions, confidence_threshold=0.3):
        """Convert predictions to readable text"""
        max_idx = np.argmax(predictions)
        confidence = predictions[max_idx]
        
        if confidence < confidence_threshold:
            return "Uncertain", confidence
        
        return self.class_names[max_idx], confidence
    
    def draw_predictions(self, frame, predictions):
        """Draw prediction results on frame"""
        pred_class, confidence = self.get_prediction_text(predictions)
        
        # Draw background rectangle for text
        cv2.rectangle(frame, (10, 10), (400, 120), (0, 0, 0), -1)
        cv2.rectangle(frame, (10, 10), (400, 120), (255, 255, 255), 2)
        
        # Draw prediction text
        cv2.putText(frame, f"Prediction: {pred_class}", 
                   (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        cv2.putText(frame, f"Confidence: {confidence:.2%}", 
                   (20, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        # Draw all class probabilities
        y_offset = 100
        for i, (class_name, prob) in enumerate(zip(self.class_names, predictions)):
            color = (0, 255, 0) if i == np.argmax(predictions) else (255, 255, 255)
            cv2.putText(frame, f"{class_name}: {prob:.2%}", 
                       (20, y_offset + i * 25), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
        
        return frame
    
    def draw_fps(self, frame):
        """Draw FPS counter"""
        self.fps_counter += 1
        elapsed_time = time.time() - self.start_time
        
        if elapsed_time >= 1.0:
            fps = self.fps_counter / elapsed_time
            self.fps_counter = 0
            self.start_time = time.time()
        else:
            fps = self.fps_counter / elapsed_time if elapsed_time > 0 else 0
        
        cv2.putText(frame, f"FPS: {fps:.1f}", 
                   (frame.shape[1] - 120, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
        
        return frame
    
    def run_inference(self, camera_index=0, window_name="Waste Classification"):
        """Run real-time inference from camera"""
        # Initialize camera
        cap = cv2.VideoCapture(camera_index)
        
        if not cap.isOpened():
            print(f"Error: Could not open camera {camera_index}")
            return
        
        # Set camera properties for better performance
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        cap.set(cv2.CAP_PROP_FPS, 30)
        
        print("Starting real-time inference...")
        print("Press 'q' to quit, 's' to save current frame")
        
        frame_count = 0
        
        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    print("Error: Could not read frame")
                    break
                
                # Preprocess frame
                processed_frame = self.preprocess_frame(frame)
                
                # Make prediction
                predictions = self.predict(processed_frame)
                
                # Draw results
                frame = self.draw_predictions(frame, predictions)
                frame = self.draw_fps(frame)
                
                # Draw instructions
                cv2.putText(frame, "Press 'q' to quit, 's' to save", 
                           (10, frame.shape[0] - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                
                # Display frame
                cv2.imshow(window_name, frame)
                
                # Handle key presses
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    break
                elif key == ord('s'):
                    # Save current frame
                    filename = f"waste_prediction_{frame_count:04d}.jpg"
                    cv2.imwrite(filename, frame)
                    print(f"Saved frame as {filename}")
                
                frame_count += 1
                
        except KeyboardInterrupt:
            print("\nStopping inference...")
        
        finally:
            cap.release()
            cv2.destroyAllWindows()
            print("Camera released and windows closed")

def main():
    # Configuration
    MODEL_PATH = "waste_classifier.tflite"  # or "waste_classifier.h5"
    CAMERA_INDEX = 0  # Usually 0 for built-in camera, 1 for external USB camera
    
    # Check if model file exists
    if not Path(MODEL_PATH).exists():
        print(f"Error: Model file '{MODEL_PATH}' not found!")
        print("Please make sure you have trained the model first.")
        return
    
    # Initialize inference
    use_tflite = MODEL_PATH.endswith('.tflite')
    inference = WasteInference(MODEL_PATH, use_tflite=use_tflite)
    
    print(f"Loaded {'TFLite' if use_tflite else 'Keras'} model: {MODEL_PATH}")
    print(f"Target classes: {inference.class_names}")
    
    # Run real-time inference
    inference.run_inference(camera_index=CAMERA_INDEX)

if __name__ == "__main__":
    main()