"""
DenseNet169 Tomato Disease Model - Optimized for Single Model Training
Based on MarkoArsenovic/DeepLearning_PlantDiseases but focused solely on DenseNet169

This module provides DenseNet169 specifically for tomato disease detection
with optimized training and inference capabilities.
"""

import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image
import io
import numpy as np


class DenseNetTomatoClassifier:
    """DenseNet169 classifier optimized for tomato disease detection"""
    
    def __init__(self):
        """Initialize DenseNet169 for 10 tomato disease classes"""
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model_type = 'densenet169'
        self.num_classes = 10
        self.input_size = (224, 224)
        
        # 10 tomato disease classes (PlantVillage subset)
        self.class_names = [
            'Tomato___Bacterial_spot',
            'Tomato___Early_blight', 
            'Tomato___Late_blight',
            'Tomato___Leaf_Mold',
            'Tomato___Septoria_leaf_spot',
            'Tomato___Spider_mites Two-spotted_spider_mite',
            'Tomato___Target_Spot',
            'Tomato___Tomato_Yellow_Leaf_Curl_Virus',
            'Tomato___Tomato_mosaic_virus',
            'Tomato___healthy'
        ]
        
        self.model = None
        
    def create_model(self):
        """Create DenseNet169 model for tomato classification"""
        # Load pre-trained DenseNet169
        model = models.densenet169(pretrained=True)
        
        # Replace classifier for tomato classes
        num_features = model.classifier.in_features
        model.classifier = nn.Linear(num_features, self.num_classes)
        
        return model
    
    def load_model(self, weights_path=None):
        """Load DenseNet169 model with optional custom weights"""
        self.model = self.create_model()
        
        if weights_path:
            # Load custom trained weights
            print(f"Loading custom weights from: {weights_path}")
            state_dict = torch.load(weights_path, map_location=self.device)
            self.model.load_state_dict(state_dict)
        else:
            # Use ImageNet pre-trained weights (already loaded)
            print("Using ImageNet pre-trained DenseNet169")
        
        self.model = self.model.to(self.device)
        self.model.eval()
        
        return self.model
    
    def get_transforms(self):
        """Get image preprocessing transforms for DenseNet169"""
        return transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
    
    def preprocess_image(self, image_data):
        """Preprocess image for DenseNet169 prediction"""
        if isinstance(image_data, bytes):
            image = Image.open(io.BytesIO(image_data))
        else:
            image = image_data
            
        # Convert to RGB if necessary
        if image.mode != 'RGB':
            image = image.convert('RGB')
            
        # Apply transforms
        transform = self.get_transforms()
        img_tensor = transform(image)
        
        # Add batch dimension
        img_tensor = img_tensor.unsqueeze(0)
        
        return img_tensor
    
    def predict(self, image):
        """Make prediction on tomato leaf image"""
        if self.model is None:
            self.load_model()
        
        # Preprocess image
        img_tensor = self.preprocess_image(image)
        img_tensor = img_tensor.to(self.device)
        
        # Make prediction
        with torch.no_grad():
            outputs = self.model(img_tensor)
            probabilities = torch.nn.functional.softmax(outputs, dim=1)
            
            # Get top prediction
            confidence, predicted_idx = torch.max(probabilities, 1)
            predicted_class = self.class_names[predicted_idx.item()]
            confidence_score = confidence.item()
            
            # Get all predictions
            all_predictions = []
            probs = probabilities.cpu().numpy()[0]
            for i, prob in enumerate(probs):
                all_predictions.append({
                    'class': self.class_names[i],
                    'confidence': float(prob)
                })
            
            # Sort by confidence
            all_predictions.sort(key=lambda x: x['confidence'], reverse=True)
        
        return {
            'predicted_class': predicted_class,
            'confidence': confidence_score,
            'all_predictions': all_predictions,
            'model_type': 'densenet169'
        }


def get_densenet_model_info():
    """Get information about the DenseNet169 tomato model"""
    return {
        'model_name': 'DenseNet169 Tomato Classifier',
        'architecture': 'DenseNet169',
        'num_classes': 10,
        'input_size': (224, 224),
        'framework': 'PyTorch',
        'pretrained_on': 'ImageNet',
        'specialized_for': 'Tomato disease detection',
        'expected_accuracy': '99%+ (with proper training)',
        'class_names': [
            'Tomato___Bacterial_spot',
            'Tomato___Early_blight', 
            'Tomato___Late_blight',
            'Tomato___Leaf_Mold',
            'Tomato___Septoria_leaf_spot',
            'Tomato___Spider_mites Two-spotted_spider_mite',
            'Tomato___Target_Spot',
            'Tomato___Tomato_Yellow_Leaf_Curl_Virus',
            'Tomato___Tomato_mosaic_virus',
            'Tomato___healthy'
        ]
    }


class TomatoDiseaseInfo:
    """Information about tomato diseases for the DenseNet model"""
    
    @staticmethod
    def get_disease_info():
        """Get detailed information about tomato diseases"""
        return {
            'Tomato___Bacterial_spot': {
                'description': 'Small, dark spots on leaves and fruit',
                'symptoms': 'Brown spots with yellow halos, fruit lesions',
                'treatment': 'Copper-based bactericides, remove infected plants',
                'severity': 'moderate'
            },
            'Tomato___Early_blight': {
                'description': 'Dark spots with concentric rings on older leaves',
                'symptoms': 'Target-like spots, yellowing leaves, stem lesions',
                'treatment': 'Fungicide application, crop rotation',
                'severity': 'moderate'
            },
            'Tomato___Late_blight': {
                'description': 'Water-soaked lesions that turn brown',
                'symptoms': 'Rapid leaf death, white mold on leaf undersides',
                'treatment': 'Preventive fungicides, remove infected plants',
                'severity': 'severe'
            },
            'Tomato___Leaf_Mold': {
                'description': 'Yellow spots on upper leaf surface',
                'symptoms': 'Fuzzy gray-green mold on leaf undersides',
                'treatment': 'Improve air circulation, reduce humidity',
                'severity': 'moderate'
            },
            'Tomato___Septoria_leaf_spot': {
                'description': 'Small circular spots with gray centers',
                'symptoms': 'Spots with dark borders, leaf yellowing',
                'treatment': 'Fungicide sprays, remove infected leaves',
                'severity': 'moderate'
            },
            'Tomato___Spider_mites Two-spotted_spider_mite': {
                'description': 'Tiny mites causing stippled leaves',
                'symptoms': 'Yellow stippling, fine webbing, leaf bronzing',
                'treatment': 'Miticides, increase humidity, predatory mites',
                'severity': 'moderate'
            },
            'Tomato___Target_Spot': {
                'description': 'Circular spots with concentric rings',
                'symptoms': 'Target-like lesions, defoliation',
                'treatment': 'Fungicide rotation, resistant varieties',
                'severity': 'moderate'
            },
            'Tomato___Tomato_Yellow_Leaf_Curl_Virus': {
                'description': 'Viral disease causing leaf curling',
                'symptoms': 'Upward leaf curling, yellowing, stunted growth',
                'treatment': 'Control whitefly vectors, resistant varieties',
                'severity': 'severe'
            },
            'Tomato___Tomato_mosaic_virus': {
                'description': 'Viral disease causing mottled leaves',
                'symptoms': 'Mosaic pattern, leaf distortion, reduced yield',
                'treatment': 'Remove infected plants, sanitize tools',
                'severity': 'severe'
            },
            'Tomato___healthy': {
                'description': 'Healthy tomato plant',
                'symptoms': 'Green, vigorous leaves with no disease signs',
                'treatment': 'Continue good cultural practices',
                'severity': 'none'
            }
        }
    
    @staticmethod
    def get_dataset_info():
        """Get dataset information for DenseNet training"""
        return {
            'dataset_name': 'PlantVillage (Tomato subset)',
            'total_classes': 10,
            'plant_type': 'Tomato',
            'image_format': 'RGB images',
            'recommended_split': {
                'train': '80%',
                'validation': '20%'
            },
            'image_requirements': {
                'format': 'JPG, PNG',
                'min_size': '224x224',
                'color_space': 'RGB'
            },
            'class_distribution': 'Balanced dataset recommended for best results'
        }