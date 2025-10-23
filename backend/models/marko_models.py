"""
Marko Model Integration  
Models from: MarkoArsenovic/DeepLearning_PlantDiseases

Contains PyTorch-based models:
- AlexNet (99.24% accuracy)
- DenseNet169 (99.72% accuracy) - Best performing
- Inception_v3 (99.76% accuracy) 
- ResNet34 (99.67% accuracy)
- VGG13 (99.49% accuracy)
- SqueezeNet1_1 (99.2% accuracy)

Supports 39 classes from PlantVillage dataset (38 diseases + 1 background)
"""

import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image
import numpy as np
import io


class MarkoPlantDiseaseClassifier:
    """Plant Disease Classification using models from Marko repository"""
    
    def __init__(self, model_type='densenet169', num_classes=39):
        """
        Initialize the classifier
        
        Args:
            model_type (str): Type of model ('alexnet', 'densenet169', 'inception_v3', 'resnet34', 'vgg13', 'squeezenet1_1')
            num_classes (int): Number of disease classes (39 for PlantVillage)
        """
        self.model_type = model_type
        self.num_classes = num_classes
        self.model = None
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # PlantVillage dataset classes (39 total: 38 diseases + 1 background)
        self.class_names = [
            'Apple___Apple_scab',
            'Apple___Black_rot', 
            'Apple___Cedar_apple_rust',
            'Apple___healthy',
            'Blueberry___healthy',
            'Cherry_(including_sour)___Powdery_mildew',
            'Cherry_(including_sour)___healthy',
            'Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot',
            'Corn_(maize)___Common_rust_',
            'Corn_(maize)___Northern_Leaf_Blight',
            'Corn_(maize)___healthy',
            'Grape___Black_rot',
            'Grape___Esca_(Black_Measles)',
            'Grape___Leaf_blight_(Isariopsis_Leaf_Spot)',
            'Grape___healthy',
            'Orange___Haunglongbing_(Citrus_greening)',
            'Peach___Bacterial_spot',
            'Peach___healthy',
            'Pepper,_bell___Bacterial_spot',
            'Pepper,_bell___healthy',
            'Potato___Early_blight',
            'Potato___Late_blight',
            'Potato___healthy',
            'Raspberry___healthy',
            'Soybean___healthy',
            'Squash___Powdery_mildew',
            'Strawberry___Leaf_scorch',
            'Strawberry___healthy',
            'Tomato___Bacterial_spot',
            'Tomato___Early_blight',
            'Tomato___Late_blight',
            'Tomato___Leaf_Mold',
            'Tomato___Septoria_leaf_spot',
            'Tomato___Spider_mites Two-spotted_spider_mite',
            'Tomato___Target_Spot',
            'Tomato___Tomato_Yellow_Leaf_Curl_Virus',
            'Tomato___Tomato_mosaic_virus',
            'Tomato___healthy',
            'Background_without_leaves'
        ]
        
        # Input sizes for different models (from repository)
        self.input_sizes = {
            'alexnet': (224, 224),
            'densenet169': (224, 224),
            'resnet34': (224, 224),
            'inception_v3': (299, 299),
            'squeezenet1_1': (224, 224),
            'vgg13': (224, 224)
        }
        
        self.img_height, self.img_width = self.input_sizes.get(model_type, (224, 224))
        
    def build_model(self):
        """Build the specified PyTorch model"""
        
        if self.model_type == 'alexnet':
            model = models.alexnet(num_classes=self.num_classes)
        elif self.model_type == 'densenet169':
            # Special handling for DenseNet169 as in repository
            model = models.DenseNet(
                num_init_features=64, 
                growth_rate=32,
                block_config=(6, 12, 32, 32), 
                num_classes=self.num_classes
            )
        elif self.model_type == 'inception_v3':
            model = models.inception_v3(num_classes=self.num_classes)
        elif self.model_type == 'resnet34':
            model = models.resnet34(num_classes=self.num_classes)
        elif self.model_type == 'vgg13':
            model = models.vgg13(num_classes=self.num_classes)
        elif self.model_type == 'squeezenet1_1':
            model = models.squeezenet1_1(num_classes=self.num_classes)
        else:
            raise ValueError(f"Unknown model type: {self.model_type}")
        
        return model
    
    def load_model(self, weights_path=None):
        """Load the model with optional pre-trained weights"""
        self.model = self.build_model()
        
        if weights_path:
            # Load custom trained weights
            state_dict = torch.load(weights_path, map_location=self.device)
            self.model.load_state_dict(state_dict)
        else:
            # Initialize with ImageNet pre-trained weights if available
            if self.model_type in ['alexnet', 'resnet34', 'vgg13', 'squeezenet1_1']:
                # Load pre-trained model and adapt final layer
                if self.model_type == 'alexnet':
                    pretrained_model = models.alexnet(pretrained=True)
                    # Replace classifier for custom number of classes
                    pretrained_model.classifier[6] = nn.Linear(
                        pretrained_model.classifier[6].in_features, 
                        self.num_classes
                    )
                elif self.model_type == 'resnet34':
                    pretrained_model = models.resnet34(pretrained=True)
                    pretrained_model.fc = nn.Linear(
                        pretrained_model.fc.in_features, 
                        self.num_classes
                    )
                elif self.model_type == 'vgg13':
                    pretrained_model = models.vgg13(pretrained=True)
                    pretrained_model.classifier[6] = nn.Linear(
                        pretrained_model.classifier[6].in_features, 
                        self.num_classes
                    )
                elif self.model_type == 'squeezenet1_1':
                    pretrained_model = models.squeezenet1_1(pretrained=True)
                    pretrained_model.classifier[1] = nn.Conv2d(
                        512, self.num_classes, kernel_size=(1,1), stride=(1,1)
                    )
                
                self.model = pretrained_model
        
        self.model.to(self.device)
        self.model.eval()
        return self.model
    
    def get_transforms(self):
        """Get image preprocessing transforms"""
        # Standard ImageNet normalization as used in repository
        normalize = transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
        
        if self.model_type == 'inception_v3':
            # Higher scale-up for inception (from repository)
            transform = transforms.Compose([
                transforms.Resize(int(max(self.img_height, self.img_width)/224*256)),
                transforms.CenterCrop(max(self.img_height, self.img_width)),
                transforms.ToTensor(),
                normalize
            ])
        else:
            transform = transforms.Compose([
                transforms.Resize(256),
                transforms.CenterCrop((self.img_height, self.img_width)),
                transforms.ToTensor(),
                normalize
            ])
        
        return transform
    
    def preprocess_image(self, image_data):
        """
        Preprocess image for prediction
        
        Args:
            image_data: Image data (PIL Image or bytes)
            
        Returns:
            Preprocessed image tensor
        """
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
        
        return img_tensor.to(self.device)
    
    def predict(self, image_data):
        """
        Make prediction on input image
        
        Args:
            image_data: Image data (PIL Image or bytes)
            
        Returns:
            Dictionary with prediction results
        """
        if self.model is None:
            self.load_model()
            
        # Preprocess image
        img_tensor = self.preprocess_image(image_data)
        
        # Make prediction
        with torch.no_grad():
            outputs = self.model(img_tensor)
            
            # Handle multiple outputs (e.g., from Inception)
            if isinstance(outputs, tuple):
                outputs = outputs[0]
            
            # Apply softmax to get probabilities
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
            'all_predictions': all_predictions[:5],  # Top 5 predictions
            'model_type': self.model_type
        }


def get_marko_model_info():
    """Get information about available Marko models"""
    return {
        'alexnet': {
            'name': 'AlexNet',
            'accuracy': '99.24%',
            'description': 'AlexNet architecture with deep retraining on PlantVillage dataset'
        },
        'densenet169': {
            'name': 'DenseNet169',
            'accuracy': '99.72%',
            'description': 'DenseNet169 - Best performing model with highest accuracy'
        },
        'inception_v3': {
            'name': 'Inception_v3',
            'accuracy': '99.76%',
            'description': 'Inception_v3 with deep retraining - Highest reported accuracy'
        },
        'resnet34': {
            'name': 'ResNet34',
            'accuracy': '99.67%',
            'description': 'ResNet34 with excellent performance and efficiency'
        },
        'vgg13': {
            'name': 'VGG13',
            'accuracy': '99.49%',
            'description': 'VGG13 architecture with deep retraining'
        },
        'squeezenet1_1': {
            'name': 'SqueezeNet1_1',
            'accuracy': '99.2%',
            'description': 'Lightweight SqueezeNet for efficient inference'
        }
    }


class MarkoDatasetInfo:
    """Information about PlantVillage dataset used in Marko repository"""
    
    @staticmethod
    def get_dataset_info():
        return {
            'name': 'PlantVillage Dataset',
            'total_classes': 39,
            'disease_classes': 38,
            'background_classes': 1,
            'plants': [
                'Apple', 'Blueberry', 'Cherry', 'Corn', 'Grape', 
                'Orange', 'Peach', 'Pepper', 'Potato', 'Raspberry',
                'Soybean', 'Squash', 'Strawberry', 'Tomato'
            ],
            'training_approach': {
                'shallow': 'Retrain only final layers',
                'deep': 'Retrain entire network', 
                'from_scratch': 'Train from random initialization'
            },
            'data_split': '80% training, 20% validation',
            'image_preprocessing': 'ImageNet normalization, various input sizes',
            'augmentation': 'Random crops, horizontal flips during training'
        }
    
    @staticmethod
    def get_plant_diseases():
        """Get diseases by plant type"""
        return {
            'Apple': ['Apple_scab', 'Black_rot', 'Cedar_apple_rust', 'healthy'],
            'Corn': ['Cercospora_leaf_spot', 'Common_rust', 'Northern_Leaf_Blight', 'healthy'],
            'Grape': ['Black_rot', 'Esca_(Black_Measles)', 'Leaf_blight', 'healthy'],
            'Tomato': [
                'Bacterial_spot', 'Early_blight', 'Late_blight', 'Leaf_Mold',
                'Septoria_leaf_spot', 'Spider_mites', 'Target_Spot', 
                'Yellow_Leaf_Curl_Virus', 'Tomato_mosaic_virus', 'healthy'
            ],
            'Potato': ['Early_blight', 'Late_blight', 'healthy'],
            'Pepper': ['Bacterial_spot', 'healthy'],
            'Others': [
                'Blueberry_healthy', 'Cherry_Powdery_mildew', 'Cherry_healthy',
                'Orange_Haunglongbing', 'Peach_Bacterial_spot', 'Peach_healthy',
                'Raspberry_healthy', 'Soybean_healthy', 'Squash_Powdery_mildew',
                'Strawberry_Leaf_scorch', 'Strawberry_healthy'
            ]
        }