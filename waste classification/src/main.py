import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import os
from pathlib import Path

class WasteClassifier:
    def __init__(self, data_dir, img_size=(224, 224), batch_size=32):
        self.data_dir = Path(data_dir)
        self.img_size = img_size
        self.batch_size = batch_size
        
        # Class mapping from original classes to 5 target classes
        # Updated to handle various possible class names
        self.class_mapping = {
            'Food Organics': 'Organic',
            'food organics': 'Organic',
            'organic': 'Organic',
            'food': 'Organic',
            'Vegetation': 'Organic',
            'vegetation': 'Organic',
            'plant': 'Organic',
            'Cardboard': 'Inorganic',
            'cardboard': 'Inorganic',
            'Glass': 'Inorganic', 
            'glass': 'Inorganic',
            'Paper': 'Inorganic',
            'paper': 'Inorganic',
            'Plastic': 'Inorganic',
            'plastic': 'Inorganic',
            'Metal': 'Metal',
            'metal': 'Metal',
            'Miscellaneous Trash': 'Electronic',
            'miscellaneous trash': 'Electronic',
            'misc': 'Electronic',
            'electronic': 'Electronic',
            'electronics': 'Electronic',
            'Textile Trash': 'Others',
            'textile trash': 'Others',
            'textile': 'Others',
            'cloth': 'Others',
            'fabric': 'Others'
        }
        
        self.target_classes = ['Organic', 'Inorganic', 'Metal', 'Electronic', 'Others']
        self.num_classes = len(self.target_classes)
        self.model = None
        
    def create_datasets(self, validation_split=0.2):
        """Create training and validation datasets with class mapping"""
        # Get original dataset
        original_train_ds = keras.utils.image_dataset_from_directory(
            self.data_dir,
            validation_split=validation_split,
            subset="training",
            seed=123,
            image_size=self.img_size,
            batch_size=self.batch_size,
            label_mode='categorical'  # Changed to categorical for mapping
        )
        
        original_val_ds = keras.utils.image_dataset_from_directory(
            self.data_dir,
            validation_split=validation_split,
            subset="validation",
            seed=123,
            image_size=self.img_size,
            batch_size=self.batch_size,
            label_mode='categorical'
        )
        
        # Get original class names
        original_classes = original_train_ds.class_names
        print(f"Original classes found: {original_classes}")
        print(f"Number of original classes: {len(original_classes)}")
        
        # Create mapping matrix
        mapping_matrix = self._create_mapping_matrix(original_classes)
        print(f"Target classes: {self.target_classes}")
        print(f"Class mapping applied successfully")
        
        # Apply mapping to datasets
        train_ds = original_train_ds.map(
            lambda x, y: (x, tf.matmul(y, mapping_matrix)),
            num_parallel_calls=tf.data.AUTOTUNE
        )
        
        val_ds = original_val_ds.map(
            lambda x, y: (x, tf.matmul(y, mapping_matrix)),
            num_parallel_calls=tf.data.AUTOTUNE
        )
        
        # Optimize datasets
        AUTOTUNE = tf.data.AUTOTUNE
        train_ds = train_ds.cache().shuffle(1000).prefetch(buffer_size=AUTOTUNE)
        val_ds = val_ds.cache().prefetch(buffer_size=AUTOTUNE)
        
        return train_ds, val_ds
    
    def _create_mapping_matrix(self, original_classes):
        """Create mapping matrix from original classes to target classes"""
        mapping_matrix = tf.zeros((len(original_classes), len(self.target_classes)))
        
        print("Class mapping details:")
        for i, orig_class in enumerate(original_classes):
            target_class = self.class_mapping.get(orig_class, 'Others')
            j = self.target_classes.index(target_class)
            mapping_matrix = tf.tensor_scatter_nd_update(
                mapping_matrix, [[i, j]], [1.0]
            )
            print(f"  {orig_class} -> {target_class}")
        
        return mapping_matrix
    
    def create_model(self):
        """Create ResNet50 model with custom head"""
        # Data augmentation
        data_augmentation = keras.Sequential([
            layers.RandomFlip("horizontal"),
            layers.RandomRotation(0.1),
            layers.RandomZoom(0.1),
        ])
        
        # Preprocessing
        preprocess_input = keras.applications.resnet50.preprocess_input
        
        # Base model
        base_model = keras.applications.ResNet50(
            weights='imagenet',
            include_top=False,
            input_shape=self.img_size + (3,)
        )
        
        # Freeze base model initially
        base_model.trainable = False
        
        # Build complete model
        inputs = keras.Input(shape=self.img_size + (3,))
        x = data_augmentation(inputs)
        x = preprocess_input(x)
        x = base_model(x, training=False)
        x = layers.GlobalAveragePooling2D()(x)
        x = layers.Dropout(0.2)(x)
        outputs = layers.Dense(self.num_classes, activation='softmax')(x)
        
        self.model = keras.Model(inputs, outputs)
        return self.model
    
    def compile_model(self, learning_rate=0.0001):
        """Compile the model"""
        self.model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
            loss='categorical_crossentropy',  # Changed to categorical
            metrics=['accuracy']
        )
    
    def train(self, train_ds, val_ds, epochs=10, fine_tune_at=100):
        """Train the model with transfer learning"""
        # Initial training
        print("Phase 1: Training classifier head...")
        history1 = self.model.fit(
            train_ds,
            epochs=epochs,
            validation_data=val_ds,
            callbacks=[
                keras.callbacks.EarlyStopping(patience=3, restore_best_weights=True),
                keras.callbacks.ReduceLROnPlateau(factor=0.2, patience=2)
            ]
        )
        
        # Fine-tuning
        print(f"Phase 2: Fine-tuning from layer {fine_tune_at}...")
        
        # Find the ResNet50 base model
        base_model = None
        for layer in self.model.layers:
            if hasattr(layer, 'layers') and layer.name == 'resnet50':
                base_model = layer
                break
        
        if base_model is None:
            print("Warning: ResNet50 base model not found. Skipping fine-tuning.")
            return history1, None
            
        base_model.trainable = True
        
        # Fine-tune from this layer onwards
        for layer in base_model.layers[:fine_tune_at]:
            layer.trainable = False
        
        # Recompile with lower learning rate
        self.model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=0.0001/10),
            loss='categorical_crossentropy',  # Changed to categorical
            metrics=['accuracy']
        )
        
        history2 = self.model.fit(
            train_ds,
            epochs=epochs,
            validation_data=val_ds,
            callbacks=[
                keras.callbacks.EarlyStopping(patience=3, restore_best_weights=True),
                keras.callbacks.ReduceLROnPlateau(factor=0.2, patience=2)
            ]
        )
        
        return history1, history2
    
    def save_model(self, save_path="waste_classifier.h5"):
        """Save the trained model"""
        self.model.save(save_path)
        print(f"Model saved to {save_path}")
    
    def convert_to_tflite(self, output_path="waste_classifier.tflite", quantize=True):
        """Convert model to TFLite format"""
        converter = tf.lite.TFLiteConverter.from_keras_model(self.model)
        
        if quantize:
            converter.optimizations = [tf.lite.Optimize.DEFAULT]
        
        tflite_model = converter.convert()
        
        with open(output_path, 'wb') as f:
            f.write(tflite_model)
        
        print(f"TFLite model saved to {output_path}")
        return output_path

def main():
    # Configuration
    DATA_DIR = "RealWaste"  # Update this path to your dataset
    IMG_SIZE = (224, 224)
    BATCH_SIZE = 32
    EPOCHS = 10
    
    # Initialize classifier
    classifier = WasteClassifier(DATA_DIR, IMG_SIZE, BATCH_SIZE)
    
    # Create datasets
    print("Creating datasets...")
    train_ds, val_ds = classifier.create_datasets()
    
    # Create and compile model
    print("Creating model...")
    classifier.create_model()
    classifier.compile_model()
    
    # Print model summary
    classifier.model.summary()
    
    # Train model
    print("Starting training...")
    history1, history2 = classifier.train(train_ds, val_ds, epochs=EPOCHS)
    
    # Save model
    classifier.save_model()
    
    # Convert to TFLite
    tflite_path = classifier.convert_to_tflite()
    
    print("Training completed!")
    print(f"Final model saved as TFLite: {tflite_path}")

if __name__ == "__main__":
    main()