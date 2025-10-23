# LeafGuard - Advanced Plant Disease Detection System

LeafGuard is an AI-powered plant disease detection system that integrates multiple state-of-the-art machine learning models for accurate plant disease identification and treatment recommendations.

## 🌟 Features

- **Multiple AI Models**: Choose from different specialized models for optimal accuracy
- **Comprehensive Disease Coverage**: Support for 39+ plant diseases across 14+ plant types
- **Real-time Analysis**: Fast inference with detailed confidence scores
- **Treatment Recommendations**: Expert advice for each identified disease
- **Modern Web Interface**: User-friendly React frontend with camera integration
- **RESTful API**: FastAPI backend with comprehensive endpoints

## 🚀 Model Integrations

This project includes two specialized model integration branches:

### 🔬 Bhargavi Models Branch (`b-model-integration`)
**Source**: [Bhargavidev26/Early-Tomato-Leaf-Disease-Prediction](https://github.com/Bhargavidev26/Early-Tomato-Leaf-Disease-Prediction)

- **Models**: Custom CNN, VGG16, ResNet50
- **Focus**: Tomato disease classification (10 classes)
- **Best Accuracy**: 94%/93% (Custom CNN)
- **Framework**: TensorFlow/Keras
- **Specialization**: Optimized for tomato crops

### 🏆 Marko Models Branch (`marko-model-integration`)
**Source**: [MarkoArsenovic/DeepLearning_PlantDiseases](https://github.com/MarkoArsenovic/DeepLearning_PlantDiseases)

- **Models**: AlexNet, DenseNet169, Inception_v3, ResNet34, VGG13, SqueezeNet1_1
- **Coverage**: PlantVillage dataset (39 classes, 14+ plant types)
- **Best Accuracy**: 99.76% (Inception_v3)
- **Framework**: PyTorch
- **Specialization**: Comprehensive multi-plant disease detection

## 📋 Project Info

**URL**: https://lovable.dev/projects/ea7323f6-27d3-42d6-9b28-90c4a79d3150

## 🛠 How to Use Different Model Branches

### Switch to Bhargavi Models (Tomato-focused)
```bash
git checkout b-model-integration
pip install -r backend/requirements.txt
uvicorn backend.main:app --host 0.0.0.0 --port 5000 --reload
```

### Switch to Marko Models (Multi-plant)
```bash
git checkout marko-model-integration
pip install -r backend/requirements.txt
uvicorn backend.main:app --host 0.0.0.0 --port 5000 --reload
```

### Available API Endpoints

#### Bhargavi Models
- `POST /predict/bhargavi/cnn` - Custom CNN model
- `POST /predict/bhargavi/vgg16` - VGG16 transfer learning
- `POST /predict/bhargavi/resnet50` - ResNet50 transfer learning

#### Marko Models
- `POST /predict/marko/densenet169` - Best performing model
- `POST /predict/marko/inception_v3` - Highest accuracy
- `POST /predict/marko/alexnet` - Fast inference
- `POST /predict/marko/resnet34` - Balanced performance
- `POST /predict/marko/vgg13` - Reliable baseline
- `POST /predict/marko/squeezenet1_1` - Lightweight model

#### Common Endpoints
- `GET /health` - System health check
- `GET /models` - Available models information
- `POST /predict` - Original model (backward compatibility)

## 🏃‍♂️ Quick Start

### Prerequisites
- Node.js & npm - [install with nvm](https://github.com/nvm-sh/nvm#installing-and-updating)
- Python 3.8+ with pip
- Git

### Installation Steps

```bash
# Step 1: Clone the repository
git clone <YOUR_GIT_URL>

# Step 2: Navigate to the project directory
cd <YOUR_PROJECT_NAME>

# Step 3: Install frontend dependencies
npm install

# Step 4: Choose your model branch
# For Bhargavi models (tomato-focused):
git checkout b-model-integration

# OR for Marko models (multi-plant):
git checkout marko-model-integration

# Step 5: Install Python dependencies
pip install -r backend/requirements.txt

# Step 6: Start the backend
uvicorn backend.main:app --host 0.0.0.0 --port 5000 --reload

# Step 7: In a new terminal, start the frontend
npm run dev
```

## 📊 Model Performance Comparison

| Model | Framework | Accuracy | Classes | Best For |
|-------|-----------|----------|---------|----------|
| Inception_v3 (Marko) | PyTorch | 99.76% | 39 | Highest accuracy |
| DenseNet169 (Marko) | PyTorch | 99.72% | 39 | Best balance |
| Custom CNN (Bhargavi) | TensorFlow | 94% | 10 | Tomato specialization |
| ResNet34 (Marko) | PyTorch | 99.67% | 39 | Efficiency |
| ResNet50 (Bhargavi) | TensorFlow | 68% | 10 | Transfer learning |

## 🔧 Development Options

## 🔧 Development Options

### Use Lovable Platform
Simply visit the [Lovable Project](https://lovable.dev/projects/ea7323f6-27d3-42d6-9b28-90c4a79d3150) and start prompting.
Changes made via Lovable will be committed automatically to this repo.

### Use Your Preferred IDE
Clone this repo and push changes. Pushed changes will also be reflected in Lovable.

### Edit Files Directly in GitHub
- Navigate to the desired file(s)
- Click the "Edit" button (pencil icon) at the top right of the file view
- Make your changes and commit the changes

### Use GitHub Codespaces
- Navigate to the main page of your repository
- Click on the "Code" button (green button) near the top right
- Select the "Codespaces" tab
- Click on "New codespace" to launch a new Codespace environment

## 🧬 Training Your Own Models

Both model branches include comprehensive training scripts:

### Bhargavi Models Training
```bash
cd backend/models
python bhargavi_trainer.py
```

### Marko Models Training
```bash
cd backend/models
python marko_trainer.py
```

## 📚 Documentation

- [Bhargavi Models Documentation](./README-Bhargavi.md) - Detailed guide for tomato disease models
- [Marko Models Documentation](./README-Marko.md) - Comprehensive multi-plant disease models

## 🤝 Contributing

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgments

- [Bhargavidev26](https://github.com/Bhargavidev26) for tomato disease classification models
- [MarkoArsenovic](https://github.com/MarkoArsenovic) for comprehensive PlantVillage models
- PlantVillage dataset contributors
- Open source community for frameworks and tools
- Edit files directly within the Codespace and commit and push your changes once you're done.

## What technologies are used for this project?

This project is built with:

- Vite
- TypeScript
- React
- shadcn-ui
- Tailwind CSS

## How can I deploy this project?

Simply open [Lovable](https://lovable.dev/projects/ea7323f6-27d3-42d6-9b28-90c4a79d3150) and click on Share -> Publish.

## Can I connect a custom domain to my Lovable project?

Yes, you can!

To connect a domain, navigate to Project > Settings > Domains and click Connect Domain.

Read more here: [Setting up a custom domain](https://docs.lovable.dev/features/custom-domain#custom-domain)
