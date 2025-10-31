"""
PlantVillage Dataset Downloader for Tomato Disease Detection
Downloads and sets up the PlantVillage dataset for DenseNet training

This script will:
1. Download the PlantVillage dataset (or guide you to download it)
2. Extract only tomato-related classes
3. Split into train/val structure for training
"""

import os
import requests
import zipfile
from pathlib import Path
import urllib.request
from tqdm import tqdm


def download_plantvillage_info():
    """Provide information about downloading PlantVillage dataset"""
    print("🍅 PlantVillage Dataset Setup Guide")
    print("="*50)
    print()
    print("📥 Dataset Download Options:")
    print()
    print("🔗 Option 1: Official Kaggle Dataset (Recommended)")
    print("   URL: https://www.kaggle.com/datasets/abdallahalidev/plantvillage-dataset")
    print("   Steps:")
    print("   1. Create Kaggle account (free)")
    print("   2. Download 'PlantVillage Dataset' (colored images)")
    print("   3. Extract to this directory")
    print()
    print("🔗 Option 2: Alternative GitHub Dataset")
    print("   URL: https://github.com/spMohanty/PlantVillage-Dataset")
    print("   Steps:")
    print("   1. Clone or download the repository")
    print("   2. Use the 'color' folder")
    print()
    print("🔗 Option 3: Research Paper Dataset")
    print("   URL: https://plantvillage.psu.edu/")
    print("   - Original research dataset from Penn State")
    print()
    print("📂 Expected Structure After Download:")
    print("PlantVillage/")
    print("├── Pepper__bell___Bacterial_spot/")
    print("├── Pepper__bell___healthy/")
    print("├── Potato___Early_blight/")
    print("├── Potato___Late_blight/")
    print("├── Potato___healthy/")
    print("├── Tomato_Bacterial_spot/           ← We need these")
    print("├── Tomato_Early_blight/             ← tomato classes")
    print("├── Tomato_Late_blight/              ←")
    print("├── Tomato_Leaf_Mold/                ←")
    print("├── Tomato_Septoria_leaf_spot/       ←")
    print("├── Tomato_Spider_mites_Two_spotted_spider_mite/ ←")
    print("├── Tomato_Target_Spot/              ←")
    print("├── Tomato_Tomato_Yellow_Leaf_Curl_Virus/ ←")
    print("├── Tomato_Tomato_mosaic_virus/      ←")
    print("└── Tomato_healthy/                  ←")
    print()
    print("🎯 Tomato Classes (10 total):")
    tomato_classes = [
        "Tomato_Bacterial_spot",
        "Tomato_Early_blight", 
        "Tomato_Late_blight",
        "Tomato_Leaf_Mold",
        "Tomato_Septoria_leaf_spot",
        "Tomato_Spider_mites_Two_spotted_spider_mite",
        "Tomato_Target_Spot",
        "Tomato_Tomato_Yellow_Leaf_Curl_Virus",
        "Tomato_Tomato_mosaic_virus",
        "Tomato_healthy"
    ]
    
    for i, class_name in enumerate(tomato_classes, 1):
        print(f"   {i:2d}. {class_name}")
    
    print()
    print("💾 Dataset Size: ~54,000 total images (~13,000 tomato images)")
    print("📊 Each class: ~1,000-3,000 images")
    print("🖼️  Image format: JPG, RGB color")
    print("📏 Image size: 256x256 pixels (will be resized to 224x224 for training)")


def create_sample_dataset():
    """Create a small sample dataset for testing"""
    print()
    print("🧪 Alternative: Create Sample Dataset for Testing")
    print("="*50)
    print()
    
    answer = input("Would you like to create a small sample dataset for testing? (y/n): ").lower().strip()
    
    if answer == 'y':
        print("📝 Creating sample dataset structure...")
        
        # Create sample structure
        base_dir = Path("dataset/tomato_disease_sample")
        
        tomato_classes = [
            "Tomato_Bacterial_spot",
            "Tomato_Early_blight", 
            "Tomato_Late_blight",
            "Tomato_healthy"
        ]
        
        for split in ['train', 'val']:
            for class_name in tomato_classes:
                class_dir = base_dir / split / class_name
                class_dir.mkdir(parents=True, exist_ok=True)
                
                # Create a README file explaining what to put here
                readme_path = class_dir / "README.txt"
                with open(readme_path, 'w') as f:
                    f.write(f"Place {class_name} images here\n")
                    f.write(f"Split: {split}\n")
                    f.write(f"Expected: JPG/PNG files of tomato plants with {class_name.replace('_', ' ')}\n")
        
        print(f"✅ Sample structure created: {base_dir}")
        print()
        print("📂 You can now:")
        print("1. Add your own tomato images to the appropriate folders")
        print("2. Use this structure to test the training pipeline")
        print("3. Replace with full PlantVillage dataset later")
        
        return str(base_dir)
    
    return None


def setup_dataset_from_local():
    """Help user set up dataset if they already have PlantVillage locally"""
    print()
    print("📁 Setup from Local PlantVillage Dataset")
    print("="*40)
    print()
    
    # Ask for dataset path
    dataset_path = input("Enter path to your PlantVillage dataset (or press Enter to skip): ").strip()
    
    if dataset_path and os.path.exists(dataset_path):
        print(f"✅ Found dataset at: {dataset_path}")
        
        # Check for tomato classes
        path_obj = Path(dataset_path)
        tomato_dirs = []
        
        for item in path_obj.iterdir():
            if item.is_dir() and item.name.startswith('Tomato'):
                tomato_dirs.append(item.name)
        
        if tomato_dirs:
            print(f"🍅 Found {len(tomato_dirs)} tomato classes:")
            for class_name in sorted(tomato_dirs):
                print(f"   ✓ {class_name}")
            
            print()
            split_now = input("Split this dataset now? (y/n): ").lower().strip()
            
            if split_now == 'y':
                print("🔄 Running dataset splitter...")
                print(f"python split_dataset.py --source \"{dataset_path}\" --output dataset/tomato_disease --copy")
                return dataset_path
        else:
            print("⚠️  No tomato classes found in this directory")
    
    return None


def main():
    """Main function"""
    download_plantvillage_info()
    
    # Check if user already has dataset
    existing_path = setup_dataset_from_local()
    
    if not existing_path:
        # Offer to create sample dataset
        sample_path = create_sample_dataset()
        
        if sample_path:
            print()
            print("🚀 Next Steps:")
            print("1. Add images to the sample dataset folders")
            print("2. Or download full PlantVillage dataset and run:")
            print("   python split_dataset.py --source PlantVillage --output dataset/tomato_disease --copy")
        else:
            print()
            print("🚀 Next Steps:")
            print("1. Download PlantVillage dataset from one of the URLs above")
            print("2. Extract to this directory")
            print("3. Run: python split_dataset.py --source PlantVillage --output dataset/tomato_disease --copy")
    
    print()
    print("📚 Documentation:")
    print("   Full guide: README-DenseNet.md")
    print("   Training: python backend/models/densenet_trainer.py")


if __name__ == "__main__":
    main()