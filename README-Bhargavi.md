# LeafGuard - Bhargavi Model Integration Branch

This branch integrates advanced tomato disease classification models from the repository [Bhargavidev26/Early-Tomato-Leaf-Disease-Prediction](https://github.com/Bhargavidev26/Early-Tomato-Leaf-Disease-Prediction).

## Integrated Models

### 1. Custom CNN Model
- **Accuracy**: 94%/93% (train/validation)
- **Architecture**: Custom CNN with 5 convolutional layers
- **Description**: Optimized CNN architecture specifically designed for tomato disease classification

### 2. VGG16 Transfer Learning Model  
- **Accuracy**: 24%/24% (train/validation)
- **Architecture**: VGG16 pre-trained + custom classifier
- **Description**: Transfer learning approach using VGG16 backbone

### 3. ResNet50 Transfer Learning Model
- **Accuracy**: 68%/73% (train/validation)  
- **Architecture**: ResNet50 pre-trained + custom classifier
- **Description**: Transfer learning approach using ResNet50 backbone

## Disease Classes

The models can detect the following tomato diseases:
- Tomato Bacterial Spot
- Tomato Early Blight
- Tomato Late Blight
- Tomato Leaf Mold
- Tomato Septoria Leaf Spot
- Tomato Spider Mites (Two-spotted Spider Mite)
- Tomato Target Spot
- Tomato Yellow Leaf Curl Virus
- Tomato Mosaic Virus
- Tomato Healthy

## API Endpoints

### Original Endpoints (Legacy)
- `GET /health` - Health check
- `POST /predict` - Predict using original model

### New Endpoints (Bhargavi Models)
- `GET /models` - Get information about available models
- `POST /predict/cnn` - Predict using Custom CNN model
- `POST /predict/vgg16` - Predict using VGG16 transfer learning model
- `POST /predict/resnet50` - Predict using ResNet50 transfer learning model

### Request Format
```json
{
  "image": "data:image/jpeg;base64,/9j/4AAQSkZJRgABAQEA..."
}
```

### Response Format
```json
{
  "is_healthy": false,
  "confidence": 94.5,
  "disease": "Tomato_Early_blight",
  "recommendations": [
    "Remove affected foliage",
    "Apply fungicide treatment", 
    "Improve air circulation"
  ],
  "timestamp": 1641234567890,
  "model_type": "cnn",
  "all_predictions": [
    {
      "class": "Tomato_Early_blight",
      "confidence": 0.945
    },
    {
      "class": "Tomato_Late_blight", 
      "confidence": 0.032
    }
  ]
}
```

## Installation & Setup

### 1. Install Dependencies
```bash
pip install tensorflow keras pillow numpy fastapi uvicorn
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

The repository includes training scripts based on the original Bhargavi implementation:

### Dataset Structure
```
dataset/
  tomato_diseases/
    Tomato_Bacterial_spot/
      img1.jpg
      img2.jpg
    Tomato_Early_blight/
      img1.jpg
      img2.jpg
    ...
```

### Training Script
```bash
cd backend/models
python bhargavi_trainer.py
```

The training script will:
- Use 80-20 train/validation split
- Apply data augmentation (rotation, shift, zoom, flip)
- Train all three models (CNN, VGG16, ResNet50)
- Save trained models in `trained_models/` directory
- Provide accuracy metrics for each model

## Model Details

### Data Preprocessing
- **Image Size**: 224x224 pixels
- **Normalization**: Pixel values scaled to [0,1]
- **Augmentation**: Rotation (20°), width/height shift (20%), shear (20%), zoom (20%), horizontal flip

### CNN Architecture
```
Conv2D(32) -> MaxPool -> Conv2D(64) -> MaxPool -> 
Conv2D(64) -> MaxPool -> Conv2D(128) -> MaxPool -> 
Conv2D(128) -> MaxPool -> Flatten -> Dense(512) -> Dense(10)
```

### Transfer Learning Approach
- Pre-trained weights from ImageNet
- Frozen base model layers
- Custom classifier head
- Global Average Pooling + Dense layers

## Performance Comparison

| Model | Train Accuracy | Val Accuracy | Best Use Case |
|-------|---------------|--------------|---------------|
| Custom CNN | 94% | 93% | Best overall performance |
| VGG16 | 24% | 24% | Baseline comparison |
| ResNet50 | 68% | 73% | Good generalization |

## Integration Features

- **Multi-model Support**: Switch between different models via API
- **Lazy Loading**: Models loaded only when first used
- **Enhanced Recommendations**: Specific treatment advice for each disease
- **Detailed Predictions**: Top-5 confidence scores for all classes
- **Backward Compatibility**: Original model still available

## Files Added/Modified

### New Files
- `backend/models/bhargavi_models.py` - Model implementations
- `backend/models/bhargavi_trainer.py` - Training script
- `README-Bhargavi.md` - This documentation

### Modified Files
- `backend/main.py` - Added new endpoints and model support
- `backend/requirements.txt` - Updated dependencies

## Future Enhancements

1. **Model Ensembling**: Combine predictions from multiple models
2. **Real-time Training**: Continuous learning from new data
3. **Model Compression**: Optimize models for edge deployment
4. **Advanced Augmentation**: More sophisticated data augmentation techniques
5. **Attention Visualization**: Show which image regions influence predictions

## References

- Original Repository: [Bhargavidev26/Early-Tomato-Leaf-Disease-Prediction](https://github.com/Bhargavidev26/Early-Tomato-Leaf-Disease-Prediction)
- TensorFlow/Keras Documentation
- Transfer Learning Best Practices

## License

This integration maintains the same license as the original LeafGuard project while respecting the licensing of the integrated Bhargavi models.