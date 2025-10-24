# LeafGuard - Advanced Plant Disease Detection System

LeafGuard is an AI-powered plant disease detection system that integrates multiple state-of-the-art machine learning models for accurate plant disease identification and treatment recommendations.

## 🌟 Features

- **Multiple AI Models**: Choose from different specialized models for optimal accuracy
- **High-Performance Detection**: Up to 99.76% accuracy with PyTorch models
- **Comprehensive Disease Coverage**: Support for 10-39 plant diseases across multiple plant types
- **Real-time Analysis**: Fast inference with detailed confidence scores
- **Treatment Recommendations**: Expert advice for each identified disease
- **Modern Web Interface**: User-friendly React frontend with camera integration
- **RESTful API**: FastAPI backend with comprehensive endpoints
- **Multiple Training Modes**: Support for transfer learning and custom training

## 🚀 Specialized Model Integration Branches

This project includes **four specialized branches**, each optimized for different use cases:

### 🍅 **Marko Tomato-Only Branch** (`marko-tomato-only`) - **⭐ RECOMMENDED**
**Source**: [MarkoArsenovic/DeepLearning_PlantDiseases](https://github.com/MarkoArsenovic/DeepLearning_PlantDiseases) (Tomato subset)

- **Models**: AlexNet, DenseNet169, Inception_v3, ResNet34, VGG13, SqueezeNet1_1
- **Focus**: Tomato disease classification only (10 classes)
- **Best Accuracy**: **99.76%** (Inception_v3)
- **Framework**: PyTorch
- **Specialization**: Highest accuracy for tomato farming applications
- **Custom Training**: Support for training with your own tomato dataset
- **Documentation**: [README-Marko-Tomato.md](https://github.com/jessuiii/leaf-wellness-app/blob/marko-tomato-only/README-Marko-Tomato.md)

### 🏆 **Marko Full Models Branch** (`marko-model-integration`)
**Source**: [MarkoArsenovic/DeepLearning_PlantDiseases](https://github.com/MarkoArsenovic/DeepLearning_PlantDiseases)

- **Models**: AlexNet, DenseNet169, Inception_v3, ResNet34, VGG13, SqueezeNet1_1
- **Coverage**: PlantVillage dataset (39 classes, 14+ plant types)
- **Best Accuracy**: 99.76% (Inception_v3)
- **Framework**: PyTorch
- **Specialization**: Comprehensive multi-plant disease detection
- **Documentation**: [README-Marko.md](https://github.com/jessuiii/leaf-wellness-app/blob/marko-model-integration/README-Marko.md)

### 🌐 **Digital Twin Integration Branch** (`digital-twin-integration`)
**Advanced Features**: Azure Digital Twins + 3D Plant Visualization

- **Azure Digital Twins**: Plant lifecycle tracking and monitoring
- **3D Visualization**: Interactive plant status grid
- **Real-time Monitoring**: Live plant health dashboard
- **Cloud Integration**: Azure Blob Storage for image management
- **IoT Ready**: Sensor data integration capabilities
- **Documentation**: [README_DigitalTwin.md](https://github.com/jessuiii/leaf-wellness-app/blob/digital-twin-integration/README_DigitalTwin.md)

### 🔬 **Bhargavi Models Branch** (`b-model-integration`)
**Source**: [Bhargavidev26/Early-Tomato-Leaf-Disease-Prediction](https://github.com/Bhargavidev26/Early-Tomato-Leaf-Disease-Prediction)

- **Models**: Custom CNN, VGG16, ResNet50
- **Focus**: Tomato disease classification (10 classes)
- **Best Accuracy**: 94% (Custom CNN)
- **Framework**: TensorFlow/Keras
- **Specialization**: Alternative approach for tomato crops
- **Documentation**: [README-Bhargavi.md](https://github.com/jessuiii/leaf-wellness-app/blob/b-model-integration/README-Bhargavi.md)

## 📊 Performance Comparison

| Branch | Best Model | Accuracy | Framework | Classes | Use Case | Training Support |
|--------|------------|----------|-----------|---------|----------|------------------|
| **marko-tomato-only** ⭐ | Inception_v3 | **99.76%** | PyTorch | 10 (tomato) | **Production farming** | ✅ Custom training |
| marko-model-integration | Inception_v3 | 99.76% | PyTorch | 39 (multi-plant) | Research & multi-crop | ✅ Custom training |
| digital-twin-integration | AI + IoT | Variable | Mixed | Variable | **Smart agriculture** | ✅ Cloud integration |
| b-model-integration | Custom CNN | 94% | TensorFlow | 10 (tomato) | Alternative approach | ✅ TensorFlow training |

## 🎯 **Recommendations by Use Case**

### 🌾 **For Tomato Farmers** → `marko-tomato-only`
- **Highest accuracy** (99.76%)
- **Tomato-focused** specialization  
- **Production-ready** with custom training support
- **6 model options** for different deployment needs

### 🔬 **For Agricultural Research** → `marko-model-integration`
- **Multi-plant coverage** (39 classes, 14+ plant types)
- **Research-grade** accuracy and flexibility
- **Comprehensive** disease knowledge base

### 🏭 **For Smart Agriculture** → `digital-twin-integration`
- **IoT integration** with Azure Digital Twins
- **Real-time monitoring** and visualization
- **Scalable** cloud infrastructure

### 🎓 **For Learning/Experimentation** → `b-model-integration`
- **TensorFlow/Keras** implementation
- **Educational** value with different approaches
- **Alternative** methodology comparison

## 🛠 Quick Start Guide

### **Switch to Your Preferred Branch**

```bash
# For tomato farming (RECOMMENDED)
git checkout marko-tomato-only
pip install -r backend/requirements.txt
uvicorn backend.main:app --host 0.0.0.0 --port 8000 --reload

# For multi-plant research
git checkout marko-model-integration  
pip install -r backend/requirements.txt
uvicorn backend.main:app --host 0.0.0.0 --port 8000 --reload

# For smart agriculture with IoT
git checkout digital-twin-integration
# Follow setup instructions in README_DigitalTwin.md

# For TensorFlow alternative
git checkout b-model-integration
pip install -r backend/requirements.txt
uvicorn backend.main:app --host 0.0.0.0 --port 5000 --reload
```

### **Frontend Development**

```bash
# Install dependencies
npm install

# Start development server
npm run dev
```

## 🧠 Training Your Own Models

### **Custom Tomato Dataset Training**

```bash
# Switch to marko-tomato-only branch
git checkout marko-tomato-only

# Configure your dataset path in backend/models/marko_trainer.py
# Update data_dir = "/path/to/your/tomato/dataset"

# Run training
cd backend/models
python marko_trainer.py
```

**Expected Dataset Structure:**
```
your_dataset/
├── train/
│   ├── Tomato___Bacterial_spot/
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

## 🌐 API Endpoints

### **Core Endpoints (All Branches)**
- `GET /health` - System health check
- `POST /predict` - Image disease prediction
- `GET /models` - Available models information

### **Marko PyTorch Models** (marko-tomato-only, marko-model-integration)
- `POST /predict/marko/{model_type}` - Predict with specific PyTorch model
- `POST /predict/marko/custom/{model_type}?weights_path=...` - Use custom trained weights
- `GET /tomato-model-info` - Tomato model specifications
- `GET /diseases` - Disease information database

### **Digital Twin Features** (digital-twin-integration)
- `GET /plants` - Digital twin plant management
- `POST /plants/{plant_id}/status` - Update plant status
- `GET /dashboard` - 3D visualization data

## 📋 Project Info

**Lovable URL**: https://lovable.dev/projects/ea7323f6-27d3-42d6-9b28-90c4a79d3150
**GitHub Repository**: https://github.com/jessuiii/leaf-wellness-app

## 🏗️ Technology Stack

### **Frontend**
- **Framework**: React + TypeScript + Vite
- **UI Components**: shadcn/ui
- **Styling**: Tailwind CSS
- **Camera Integration**: HTML5 Media API

### **Backend**
- **API Framework**: FastAPI (Python)
- **ML Frameworks**: PyTorch, TensorFlow/Keras
- **Image Processing**: PIL, OpenCV
- **Database**: JSON-based (expandable)

### **Cloud & DevOps** (digital-twin branch)
- **Cloud Platform**: Microsoft Azure
- **Digital Twins**: Azure Digital Twins
- **Storage**: Azure Blob Storage
- **Deployment**: Docker-ready

## 🎓 Getting Started

1. **Choose your branch** based on your use case (see recommendations above)
2. **Clone the repository** and switch to your chosen branch
3. **Install dependencies** (Python backend + Node.js frontend)
4. **Configure your dataset** if training custom models
5. **Start the servers** and begin detecting plant diseases!

## 🤝 Contributing

Each branch serves different purposes - contribute to the branch that matches your expertise:
- **PyTorch models** → marko branches
- **TensorFlow models** → bhargavi branch  
- **IoT/Cloud features** → digital-twin branch
- **Frontend improvements** → any branch

## 📄 License

This project integrates multiple open-source repositories. Please check individual branch documentation for specific license information.

---

**🌱 Happy Plant Disease Detection!** Choose your branch and start protecting crops with AI! 🚀
