# 🍅 LeafGuard - DenseNet169 Tomato Disease Detection

**Specialized branch focused exclusively on DenseNet169 for optimal tomato disease detection**

## 🎯 Why This Branch?

This branch is dedicated to **DenseNet169**, the **best performing single model** for tomato disease detection:

- **🏆 Best Accuracy**: 99.72% on PlantVillage dataset
- **⚡ Optimal Performance**: Perfect balance of speed and accuracy  
- **🔧 Simplified Training**: Single model focus for easier deployment
- **📦 Production Ready**: Streamlined for agricultural applications

## 🚀 Features

### **Core Capabilities**
- **DenseNet169 Architecture**: State-of-the-art CNN for image classification
- **10 Tomato Diseases**: Comprehensive tomato-specific disease detection
- **Custom Training**: Train with your own tomato datasets
- **Custom Weights**: Load and use your trained models
- **REST API**: FastAPI backend with specialized endpoints

### **Model Performance**
- **Training Accuracy**: 95-99%+
- **Inference Speed**: ~50ms per image
- **Model Size**: ~28MB
- **Memory Usage**: Optimized for production deployment

## 📊 Supported Tomato Diseases

1. **Tomato___Bacterial_spot** - Bacterial infection with dark spots
2. **Tomato___Early_blight** - Fungal disease with target-like spots  
3. **Tomato___Late_blight** - Devastating water-soaked lesions
4. **Tomato___Leaf_Mold** - Yellow spots with gray-green mold
5. **Tomato___Septoria_leaf_spot** - Small circular spots
6. **Tomato___Spider_mites Two-spotted_spider_mite** - Mite damage
7. **Tomato___Target_Spot** - Concentric ring lesions
8. **Tomato___Tomato_Yellow_Leaf_Curl_Virus** - Viral leaf curling
9. **Tomato___Tomato_mosaic_virus** - Viral mosaic patterns
10. **Tomato___healthy** - Healthy plant detection

## 🛠 Quick Start

### **1. Setup Environment**

```bash
# Clone and switch to densenet branch
git clone https://github.com/jessuiii/leaf-wellness-app.git
cd leaf-wellness-app
git checkout densenet-tomato-only

# Install dependencies
pip install torch torchvision pillow fastapi uvicorn numpy
```

### **2. Start the API Server**

```bash
# Start DenseNet API server
uvicorn backend.main:app --host 0.0.0.0 --port 8000 --reload
```

### **3. Test the Model**

```bash
# Test with DenseNet169 (primary endpoint)
curl -X POST "http://localhost:8000/predict/densenet" \
     -H "Content-Type: application/json" \
     -d '{"image": "data:image/jpeg;base64,..."}'

# Check model info
curl "http://localhost:8000/densenet-info"
```

## 🏋️ Training Your Own DenseNet Model

### **Dataset Preparation**

Organize your tomato dataset like this:

```
dataset/tomato_disease/
├── train/
│   ├── Tomato___Bacterial_spot/
│   │   ├── image1.jpg
│   │   ├── image2.jpg
│   │   └── ...
│   ├── Tomato___Early_blight/
│   ├── Tomato___Late_blight/
│   ├── Tomato___Leaf_Mold/
│   ├── Tomato___Septoria_leaf_spot/
│   ├── Tomato___Spider_mites Two-spotted_spider_mite/
│   ├── Tomato___Target_Spot/
│   ├── Tomato___Tomato_Yellow_Leaf_Curl_Virus/
│   ├── Tomato___Tomato_mosaic_virus/
│   └── Tomato___healthy/
└── val/
    └── (same structure as train)
```

### **Run Training**

```bash
# Navigate to models directory
cd backend/models

# Update dataset path in densenet_trainer.py
# DATASET_PATH = "/path/to/your/tomato_disease"

# Start training
python densenet_trainer.py
```

### **Training Output**

```
🍅 DenseNet169 Tomato Disease Trainer
==================================================
📁 Dataset: dataset/tomato_disease
💾 Save dir: trained_models
🔧 Device: cuda

📂 Loading tomato dataset...
✅ Dataset loaded!
📊 Train: 8000, Val: 2000

🏋️ Training DenseNet169...
📅 Epoch 1/25
TRAIN Loss: 1.2345 Acc: 0.7890
VAL Loss: 0.8765 Acc: 0.8234
🎉 New best! Accuracy: 0.8234

...

🏁 Training complete in 180m 45s
🏆 Best val accuracy: 0.9876
💾 Model saved: trained_models/densenet169_tomato.pth
```

### **Use Your Trained Model**

```bash
# Test with your custom trained model
curl -X POST "http://localhost:8000/predict/densenet/custom" \
     -H "Content-Type: application/json" \
     -d '{"image": "data:image/jpeg;base64,...", "weights_path": "trained_models/densenet169_tomato.pth"}'
```

## 🔧 API Endpoints

### **Primary Endpoints**

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/predict/densenet` | POST | **Primary prediction endpoint** |
| `/predict/densenet/custom` | POST | Predict with custom trained weights |
| `/densenet-info` | GET | Model information and capabilities |
| `/densenet-diseases` | GET | Tomato disease database |
| `/densenet-dataset` | GET | Dataset and training information |

### **Legacy Support**

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/predict` | POST | Original model (backup) |
| `/predict/marko/{model_type}` | POST | Marko models (if available) |
| `/health` | GET | System health check |

## 📖 Programming Interface

### **Basic Usage**

```python
from backend.models.densenet_tomato_model import DenseNetTomatoClassifier
from PIL import Image

# Create classifier
classifier = DenseNetTomatoClassifier()

# Load model (ImageNet pretrained)
classifier.load_model()

# Make prediction
image = Image.open('tomato_leaf.jpg')
result = classifier.predict(image)

print(f"Disease: {result['predicted_class']}")
print(f"Confidence: {result['confidence']:.4f}")
```

### **Custom Trained Model**

```python
# Load your custom trained model
classifier = DenseNetTomatoClassifier()
classifier.load_model(weights_path='trained_models/densenet169_tomato.pth')

# Use for prediction
result = classifier.predict(image)
```

### **Training Script**

```python
from backend.models.densenet_trainer import DenseNetTrainer

# Initialize trainer
trainer = DenseNetTrainer(data_dir="dataset/tomato_disease")

# Run complete training pipeline
model_path = trainer.run_training()
print(f"Trained model saved: {model_path}")
```

## 🎛 Configuration

### **Training Parameters**

```python
# In densenet_trainer.py
self.batch_size = 32      # Adjust based on GPU memory
self.epochs = 25          # Increase for better accuracy
self.learning_rate = 0.001
self.momentum = 0.9
```

### **Model Parameters**

```python
# In densenet_tomato_model.py
self.num_classes = 10     # Fixed for tomato diseases
self.input_size = (224, 224)  # DenseNet standard
```

## 📈 Performance Optimization

### **For Training**
- **GPU Recommended**: Use CUDA for 10x speed improvement
- **Batch Size**: Increase if you have more GPU memory
- **Data Augmentation**: Enabled by default for better generalization
- **Learning Rate Scheduling**: Automatic decay for optimal convergence

### **For Inference**
- **Model Caching**: Models loaded once and reused
- **Batch Processing**: Process multiple images together
- **GPU Acceleration**: Automatic CUDA usage when available

## 🐛 Troubleshooting

### **Common Issues**

1. **Import Errors**
   ```bash
   pip install torch torchvision pillow fastapi uvicorn
   ```

2. **CUDA Out of Memory**
   ```python
   # Reduce batch size in densenet_trainer.py
   self.batch_size = 16  # Instead of 32
   ```

3. **Dataset Not Found**
   ```python
   # Update path in densenet_trainer.py
   DATASET_PATH = "/full/path/to/your/dataset"
   ```

4. **Low Training Accuracy**
   - Check data quality and balance
   - Increase training epochs
   - Verify class names match expected format

## 📋 System Requirements

### **Minimum Requirements**
- **Python**: 3.8+
- **RAM**: 8GB+
- **Storage**: 5GB+ free space
- **CPU**: Multi-core recommended

### **Recommended for Training**
- **GPU**: NVIDIA GPU with 8GB+ VRAM
- **RAM**: 16GB+
- **Storage**: 20GB+ free space
- **CPU**: High-performance multi-core

## 🔗 Branch Comparison

| Feature | DenseNet Branch | Marko-Tomato Branch | Full Marko Branch |
|---------|----------------|-------------------|------------------|
| **Primary Model** | DenseNet169 only | 6 PyTorch models | 6 PyTorch models |
| **Classes** | 10 (tomato) | 10 (tomato) | 39 (multi-plant) |
| **Simplicity** | ⭐⭐⭐⭐⭐ | ⭐⭐⭐ | ⭐⭐ |
| **Performance** | 99.72% | 99.76% | 99.76% |
| **Training Speed** | Fast | Medium | Slow |
| **Use Case** | Production deployment | Research comparison | Multi-plant research |

## 🎯 When to Use This Branch

### **✅ Perfect For:**
- **Production tomato farming** applications
- **Single model deployment** scenarios  
- **Custom training** with your own datasets
- **Performance-critical** applications
- **Educational purposes** for learning DenseNet

### **❌ Consider Other Branches For:**
- **Multi-plant** disease detection → Use `marko-model-integration`
- **Model comparison** research → Use `marko-tomato-only`
- **IoT/Cloud integration** → Use `digital-twin-integration`

## 🤝 Contributing

1. **Fork the repository**
2. **Create feature branch**: `git checkout -b feature/amazing-densenet-feature`
3. **Focus on DenseNet**: Keep changes related to DenseNet169 optimization
4. **Test thoroughly**: Ensure training and inference work correctly
5. **Submit PR**: Detailed description of DenseNet improvements

## 📄 License

This project integrates and builds upon multiple open-source repositories. See individual files for specific license information.

---

## 🏆 **Ready to Detect Tomato Diseases with DenseNet169?**

**🍅 Start with the best-performing single model for tomato disease detection!**

```bash
git checkout densenet-tomato-only
uvicorn backend.main:app --host 0.0.0.0 --port 8000 --reload
```

**🚀 Happy Tomato Disease Detection with DenseNet169!** 🌱