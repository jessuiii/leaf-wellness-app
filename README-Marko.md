# LeafGuard - Marko Model Integration Branch

This branch integrates advanced PyTorch-based plant disease classification models from the repository [MarkoArsenovic/DeepLearning_PlantDiseases](https://github.com/MarkoArsenovic/DeepLearning_PlantDiseases).

## Integrated Models

### PyTorch Models from Marko Repository

1. **AlexNet** - Accuracy: 99.24%
2. **DenseNet169** - Accuracy: 99.72% ⭐ (Best performing)
3. **Inception_v3** - Accuracy: 99.76% ⭐ (Highest accuracy)
4. **ResNet34** - Accuracy: 99.67%
5. **VGG13** - Accuracy: 99.49%
6. **SqueezeNet1_1** - Accuracy: 99.2% (Lightweight)

### Training Modes
- **Shallow**: Retrain only final layers (fast)
- **Deep**: Retrain entire network (best performance)
- **From Scratch**: Train from random initialization

## PlantVillage Dataset (39 Classes)

The models classify diseases across multiple plant types:

### Supported Plants:
- **Apple**: Apple scab, Black rot, Cedar apple rust, Healthy
- **Corn**: Cercospora leaf spot, Common rust, Northern Leaf Blight, Healthy
- **Grape**: Black rot, Esca (Black Measles), Leaf blight, Healthy
- **Tomato**: 10 classes including Bacterial spot, Early/Late blight, Leaf Mold, etc.
- **Potato**: Early blight, Late blight, Healthy
- **Other plants**: Blueberry, Cherry, Orange, Peach, Pepper, Raspberry, Soybean, Squash, Strawberry

## API Endpoints

### Health & Info
- `GET /health` - System health check
- `GET /models` - Get available models information
- `GET /dataset` - PlantVillage dataset information
- `GET /diseases` - Diseases organized by plant type

### Prediction Endpoints
- `POST /predict` - Original model (legacy/backup)
- `POST /predict/marko/{model_type}` - Predict using specific Marko model

### Available Model Types:
- `alexnet`
- `densenet169` (recommended)
- `inception_v3` (highest accuracy)
- `resnet34`
- `vgg13`
- `squeezenet1_1`

## Request/Response Format

### Request
```json
{
  "image": "data:image/jpeg;base64,/9j/4AAQSkZJRgABAQEA..."
}
```

### Response
```json
{
  "is_healthy": false,
  "confidence": 99.72,
  "disease": "Tomato___Early_blight",
  "recommendations": [
    "Remove affected foliage",
    "Apply fungicide treatment",
    "Improve air circulation"
  ],
  "timestamp": 1641234567890,
  "model_type": "marko_densenet169",
  "all_predictions": [
    {
      "class": "Tomato___Early_blight",
      "confidence": 0.9972
    },
    {
      "class": "Tomato___Late_blight",
      "confidence": 0.0015
    }
  ]
}
```

## Installation & Setup

### 1. Install Dependencies
```bash
pip install -r backend/requirements.txt
```

### 2. Start the Backend
```bash
# From the project root
uvicorn backend.main:app --host 0.0.0.0 --port 5000 --reload
```

### 3. Start the Frontend
```bash
npm install
npm run dev
```

## Training Your Own Models

The repository includes comprehensive training scripts based on the original Marko implementation:

### Dataset Structure
```
dataset/
  PlantVillage/
    train/
      Apple___Apple_scab/
        img1.jpg
        img2.jpg
      Apple___Black_rot/
        img1.jpg
        img2.jpg
      ...
    val/
      Apple___Apple_scab/
        img1.jpg
        img2.jpg
      ...
```

### Training Script
```bash
cd backend/models
python marko_trainer.py
```

The training script will:
- Train all 6 model architectures
- Test 3 training modes each (shallow, deep, from_scratch)
- Use ImageNet pre-trained weights for transfer learning
- Apply data augmentation (random crops, horizontal flips)
- Save trained models and statistics
- Generate results CSV file

## Model Performance Comparison

| Model | Training Mode | Accuracy | Best Use Case |
|-------|--------------|----------|---------------|
| Inception_v3 | Deep | 99.76% | Highest accuracy |
| DenseNet169 | Deep | 99.72% | Best balance |
| ResNet34 | Deep | 99.67% | Good efficiency |
| VGG13 | Deep | 99.49% | Reliable baseline |
| AlexNet | Deep | 99.24% | Fast inference |
| SqueezeNet1_1 | Deep | 99.2% | Mobile/edge devices |

## Technical Details

### Data Preprocessing
- **Image Sizes**: 224x224 (most models), 299x299 (Inception_v3)
- **Normalization**: ImageNet standards ([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
- **Augmentation**: Random crops, horizontal flips during training

### Model Architectures
- **Transfer Learning**: ImageNet pre-trained weights
- **Custom Classifiers**: Adapted for 39 PlantVillage classes
- **Optimization**: SGD with lr=0.001, momentum=0.9
- **Training**: 15 epochs with early stopping

### Hardware Support
- **GPU**: CUDA support for faster training/inference
- **CPU**: Fallback for systems without GPU
- **Memory**: Configurable GPU memory usage

## Integration Features

- **Multi-model Support**: 6 different PyTorch architectures
- **Lazy Loading**: Models loaded only when first used
- **Comprehensive Disease Coverage**: 39 classes across 14 plant types
- **Expert Recommendations**: Disease-specific treatment advice
- **Detailed Predictions**: Top-5 confidence scores
- **Dataset Information**: Complete PlantVillage dataset details
- **Backward Compatibility**: Original TensorFlow model still available

## Files Added/Modified

### New Files
- `backend/models/marko_models.py` - PyTorch model implementations
- `backend/models/marko_trainer.py` - Training script with all modes
- `README-Marko.md` - This documentation

### Modified Files
- `backend/main.py` - Enhanced with Marko model endpoints
- `backend/requirements.txt` - Added PyTorch dependencies

## Example Usage

### Get Available Models
```bash
curl http://localhost:5000/models
```

### Predict with DenseNet169
```bash
curl -X POST http://localhost:5000/predict/marko/densenet169 \
  -H "Content-Type: application/json" \
  -d '{"image": "data:image/jpeg;base64,..."}'
```

### Get Dataset Information
```bash
curl http://localhost:5000/dataset
```

## Future Enhancements

1. **Model Ensembling**: Combine predictions from multiple models
2. **Attention Visualization**: Show which image regions influence predictions
3. **Real-time Training**: Continuous learning from new data
4. **Model Quantization**: Optimize models for mobile deployment
5. **Saliency Maps**: Visual explanations of model decisions
6. **Occlusion Analysis**: Understanding model focus areas

## References

- Original Repository: [MarkoArsenovic/DeepLearning_PlantDiseases](https://github.com/MarkoArsenovic/DeepLearning_PlantDiseases)
- PlantVillage Dataset
- PyTorch Documentation
- Transfer Learning Best Practices

## License

This integration maintains the same license as the original LeafGuard project while respecting the licensing of the integrated Marko models.