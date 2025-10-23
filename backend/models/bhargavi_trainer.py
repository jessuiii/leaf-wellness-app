"""
Training Script for Bhargavi Models
Based on: Bhargavidev26/Early-Tomato-Leaf-Disease-Prediction

This script implements the training approach from the Bhargavi repository
with data augmentation and multiple model architectures.
"""

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import os
import numpy as np
from bhargavi_models import BhargaviTomatoDiseaseClassifier


class BhargaviModelTrainer:
    """Training pipeline for Bhargavi tomato disease models"""
    
    def __init__(self, data_dir, model_type='cnn'):
        """
        Initialize trainer
        
        Args:
            data_dir (str): Path to dataset directory
            model_type (str): Type of model to train ('cnn', 'vgg16', 'resnet50')
        """
        self.data_dir = data_dir
        self.model_type = model_type
        self.img_height = 224
        self.img_width = 224
        self.batch_size = 32
        self.epochs = 50
        
        # Class names from the repository
        self.class_names = [
            'Tomato_Bacterial_spot',
            'Tomato_Early_blight', 
            'Tomato_Late_blight',
            'Tomato_Leaf_Mold',
            'Tomato_Septoria_leaf_spot',
            'Tomato_Spider_mites_Two_spotted_spider_mite',
            'Tomato__Target_Spot',
            'Tomato__Tomato_YellowLeaf__Curl_Virus',
            'Tomato__Tomato_mosaic_virus',
            'Tomato_healthy'
        ]
        
    def create_data_generators(self):
        """Create data generators with augmentation (based on repository approach)"""
        
        # Training data generator with augmentation
        train_datagen = ImageDataGenerator(
            rescale=1./255,
            rotation_range=20,
            width_shift_range=0.2,
            height_shift_range=0.2,
            shear_range=0.2,
            zoom_range=0.2,
            horizontal_flip=True,
            validation_split=0.2  # 80-20 split as in repository
        )
        
        # Validation data generator (only rescaling)
        val_datagen = ImageDataGenerator(
            rescale=1./255,
            validation_split=0.2
        )
        
        # Create generators
        train_generator = train_datagen.flow_from_directory(
            self.data_dir,
            target_size=(self.img_height, self.img_width),
            batch_size=self.batch_size,
            class_mode='categorical',
            subset='training',
            classes=self.class_names
        )
        
        validation_generator = val_datagen.flow_from_directory(
            self.data_dir,
            target_size=(self.img_height, self.img_width),
            batch_size=self.batch_size,
            class_mode='categorical',
            subset='validation',
            classes=self.class_names
        )
        
        return train_generator, validation_generator
    
    def train_model(self, save_path=None):
        """
        Train the specified model
        
        Args:
            save_path (str): Path to save the trained model
            
        Returns:
            Trained model and training history
        """
        print(f"Training {self.model_type} model...")
        
        # Create model
        classifier = BhargaviTomatoDiseaseClassifier(model_type=self.model_type)
        model = classifier.load_model()
        
        # Create data generators
        train_generator, validation_generator = self.create_data_generators()
        
        # Callbacks
        callbacks = [
            keras.callbacks.EarlyStopping(
                monitor='val_accuracy',
                patience=10,
                restore_best_weights=True
            ),
            keras.callbacks.ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.2,
                patience=5,
                min_lr=0.0001
            )
        ]
        
        if save_path:
            callbacks.append(
                keras.callbacks.ModelCheckpoint(
                    save_path,
                    monitor='val_accuracy',
                    save_best_only=True,
                    verbose=1
                )
            )
        
        # Train model
        history = model.fit(
            train_generator,
            epochs=self.epochs,
            validation_data=validation_generator,
            callbacks=callbacks,
            verbose=1
        )
        
        # Evaluate model
        print("\nEvaluating model on validation data...")
        val_loss, val_accuracy = model.evaluate(validation_generator)
        print(f"Validation Loss: {val_loss:.4f}")
        print(f"Validation Accuracy: {val_accuracy:.4f}")
        
        return model, history
    
    def train_all_models(self, save_dir="trained_models"):
        """Train all three model types and save them"""
        
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        
        results = {}
        
        for model_type in ['cnn', 'vgg16', 'resnet50']:
            print(f"\n{'='*50}")
            print(f"Training {model_type.upper()} Model")
            print(f"{'='*50}")
            
            self.model_type = model_type
            save_path = os.path.join(save_dir, f"bhargavi_{model_type}_model.h5")
            
            try:
                model, history = self.train_model(save_path)
                
                # Store results
                final_val_accuracy = max(history.history['val_accuracy'])
                final_train_accuracy = max(history.history['accuracy'])
                
                results[model_type] = {
                    'model': model,
                    'history': history,
                    'final_train_accuracy': final_train_accuracy,
                    'final_val_accuracy': final_val_accuracy,
                    'save_path': save_path
                }
                
                print(f"{model_type.upper()} - Train Accuracy: {final_train_accuracy:.4f}")
                print(f"{model_type.upper()} - Val Accuracy: {final_val_accuracy:.4f}")
                
            except Exception as e:
                print(f"Error training {model_type}: {str(e)}")
                results[model_type] = {'error': str(e)}
        
        return results


def main():
    """Main training function"""
    # Set dataset path (update this to your dataset location)
    data_dir = "dataset/tomato_diseases"  # Update this path
    
    if not os.path.exists(data_dir):
        print(f"Dataset directory not found: {data_dir}")
        print("Please update the data_dir path to point to your tomato disease dataset")
        return
    
    # Create trainer
    trainer = BhargaviModelTrainer(data_dir)
    
    # Train all models
    results = trainer.train_all_models()
    
    # Print summary
    print(f"\n{'='*60}")
    print("TRAINING SUMMARY")
    print(f"{'='*60}")
    
    for model_type, result in results.items():
        if 'error' in result:
            print(f"{model_type.upper()}: FAILED - {result['error']}")
        else:
            print(f"{model_type.upper()}:")
            print(f"  Train Accuracy: {result['final_train_accuracy']:.4f}")
            print(f"  Val Accuracy: {result['final_val_accuracy']:.4f}")
            print(f"  Saved to: {result['save_path']}")
            print()


if __name__ == "__main__":
    main()