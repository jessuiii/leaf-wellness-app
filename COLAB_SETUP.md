# 🚀 Google Colab Setup Guide - DenseNet169 Training

## 📋 **Quick Setup Checklist**

### **Files to Upload to Colab:**
1. ✅ `densenet_trainer.py` (from `backend/models/`)
2. ✅ `densenet_tomato_model.py` (from `backend/models/`)  
3. ✅ `DenseNet_Colab_Training.ipynb` (this notebook)
4. 📁 Your dataset: `dataset/tomato_disease/` (zipped)

### **🔬 Optional Visualization Files:**
5. ✅ `densenet_saliency.py` - Generate saliency maps
6. ✅ `densenet_occlusion.py` - Occlusion analysis
7. ✅ `densenet_plot.py` - Training progress plots
8. ✅ `torchvis_util.py` - Visualization utilities

**Note**: Upload visualization files for model interpretability and analysis!

### **⚡ Quick Start:**
1. Open Google Colab → Upload the notebook
2. Upload the 2 Python files to Colab
3. Upload your dataset (zip it first)
4. Run all cells in order
5. Download trained model when complete

## 🔧 **Detailed Instructions**

### **Step 1: Prepare Files Locally**
```bash
# Navigate to your project
cd leaf-wellness-app

# Copy essential files to a Colab folder
mkdir colab_training
cp backend/models/densenet_trainer.py colab_training/
cp backend/models/densenet_tomato_model.py colab_training/
cp DenseNet_Colab_Training.ipynb colab_training/

# Copy visualization files for model analysis
cp backend/densenet_saliency.py colab_training/
cp backend/densenet_occlusion.py colab_training/
cp backend/densenet_plot.py colab_training/
cp backend/visualization/torchvis_util.py colab_training/

# Zip your dataset (exclude this from git)
# Your dataset should be at: dataset/tomato_disease/
zip -r tomato_disease.zip dataset/tomato_disease/
```

### **Step 2: Upload to Google Colab**
1. Go to [Google Colab](https://colab.research.google.com/)
2. **Upload Notebook**: `DenseNet_Colab_Training.ipynb`
3. **Upload Files** (drag & drop to file panel):
   - `densenet_trainer.py`
   - `densenet_tomato_model.py`
   - `densenet_saliency.py` (optional)
   - `densenet_occlusion.py` (optional)
   - `densenet_plot.py` (optional)
   - `torchvis_util.py` (optional)
   - `tomato_disease.zip` (your dataset)

### **Step 3: Enable GPU**
1. Runtime → Change runtime type
2. Hardware accelerator → **GPU**
3. GPU type → **T4** (free tier)

### **Step 4: Run Training**
1. Run cells in order (Ctrl+Enter)
2. Training takes **30-90 minutes** on GPU
3. Monitor progress in output

### **Step 5: Download Results**
1. Download `densenet169_tomato.pth` (trained model)
2. Place in your local `trained_models/` folder
3. Update your local API to use the new model

## 📊 **Expected Results**

| Metric | CPU (Local) | GPU (Colab) |
|--------|-------------|-------------|
| **Time** | 6-12 hours | 30-90 min |
| **Accuracy** | 99.72% | 99.72% |
| **Batch Size** | 8 (memory limited) | 32+ (optimized) |
| **Epochs** | 25 | 25 |
| **Visualizations** | Limited | ✅ Saliency + Occlusion |

## 🔍 **Troubleshooting**

### **Common Issues:**
1. **"Module not found"** → Ensure files are uploaded to Colab root
2. **"Dataset not found"** → Check zip extraction path
3. **"CUDA out of memory"** → Reduce batch size in trainer
4. **Slow training** → Check GPU is enabled

### **Dataset Requirements:**
- ✅ 10 tomato disease classes
- ✅ Train/val split (80/20)
- ✅ Images in correct folder structure
- ✅ Total: ~32,022 images

## 🌐 **Alternative: GitHub Clone**
```python
# In Colab, clone your repo directly
!git clone https://github.com/jessuiii/leaf-wellness-app.git
!cd leaf-wellness-app && git checkout densenet-tomato-only
# Then upload just your dataset
```

## 💡 **Pro Tips:**
- 🔋 **Keep Colab active** - move mouse occasionally during training
- 💾 **Save frequently** - download checkpoints if training is long
- 🚀 **Use GPU Pro** - for faster training with better GPUs
- 📊 **Monitor logs** - watch for overfitting or errors

---

**🍅 Ready to train your DenseNet model with GPU power!** 🚀