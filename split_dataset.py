"""
Dataset Splitter for PlantVillage Tomato Data
Splits the PlantVillage dataset into train/val structure for DenseNet training

Usage:
    python split_dataset.py --source PlantVillage --output dataset/tomato_disease --split 0.8

This will:
1. Filter only tomato classes from PlantVillage
2. Create train/val split (80% train, 20% val by default)
3. Copy files to the new structure expected by DenseNet trainer
"""

import os
import shutil
import argparse
from pathlib import Path
import random


def parse_arguments():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='Split PlantVillage dataset for DenseNet training')
    parser.add_argument('--source', required=True, help='Path to PlantVillage dataset directory')
    parser.add_argument('--output', default='dataset/tomato_disease', help='Output directory for split dataset')
    parser.add_argument('--split', type=float, default=0.8, help='Train split ratio (default: 0.8)')
    parser.add_argument('--seed', type=int, default=42, help='Random seed for reproducible splits')
    parser.add_argument('--copy', action='store_true', help='Copy files instead of moving them')
    
    return parser.parse_args()


def get_tomato_classes():
    """Get list of tomato disease classes"""
    return [
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


def split_class_data(source_dir, output_dir, class_name, split_ratio, copy_files=False):
    """
    Split a single class directory into train/val
    
    Args:
        source_dir: Source directory with images
        output_dir: Output base directory  
        class_name: Disease class name
        split_ratio: Ratio for train split
        copy_files: Whether to copy (True) or move (False) files
    """
    # Get all image files
    source_path = Path(source_dir)
    image_files = []
    
    for ext in ['*.jpg', '*.jpeg', '*.png', '*.JPG', '*.JPEG', '*.PNG']:
        image_files.extend(list(source_path.glob(ext)))
    
    if not image_files:
        print(f"⚠️  No images found in {source_dir}")
        return 0, 0
    
    # Shuffle for random split
    random.shuffle(image_files)
    
    # Calculate split point
    split_point = int(len(image_files) * split_ratio)
    train_files = image_files[:split_point]
    val_files = image_files[split_point:]
    
    # Create output directories
    train_dir = Path(output_dir) / 'train' / class_name
    val_dir = Path(output_dir) / 'val' / class_name
    
    train_dir.mkdir(parents=True, exist_ok=True)
    val_dir.mkdir(parents=True, exist_ok=True)
    
    # Copy/move train files
    for file_path in train_files:
        dest_path = train_dir / file_path.name
        if copy_files:
            shutil.copy2(file_path, dest_path)
        else:
            shutil.move(str(file_path), str(dest_path))
    
    # Copy/move val files
    for file_path in val_files:
        dest_path = val_dir / file_path.name
        if copy_files:
            shutil.copy2(file_path, dest_path)
        else:
            shutil.move(str(file_path), str(dest_path))
    
    return len(train_files), len(val_files)


def main():
    """Main function"""
    args = parse_arguments()
    
    # Set random seed
    random.seed(args.seed)
    
    print("🍅 PlantVillage Dataset Splitter for DenseNet Training")
    print("="*60)
    print(f"📂 Source: {args.source}")
    print(f"📁 Output: {args.output}")
    print(f"📊 Split ratio: {args.split:.1%} train, {1-args.split:.1%} val")
    print(f"🎲 Random seed: {args.seed}")
    print(f"🔄 Mode: {'Copy' if args.copy else 'Move'} files")
    print()
    
    # Check source directory
    source_path = Path(args.source)
    if not source_path.exists():
        print(f"❌ Source directory not found: {args.source}")
        return
    
    # Get tomato classes
    tomato_classes = get_tomato_classes()
    print(f"🎯 Looking for {len(tomato_classes)} tomato disease classes...")
    
    # Create output directory
    output_path = Path(args.output)
    output_path.mkdir(parents=True, exist_ok=True)
    
    total_train = 0
    total_val = 0
    processed_classes = 0
    
    # Process each tomato class
    for class_name in tomato_classes:
        class_dir = source_path / class_name
        
        if class_dir.exists() and class_dir.is_dir():
            print(f"📦 Processing {class_name}...")
            train_count, val_count = split_class_data(
                class_dir, args.output, class_name, args.split, args.copy
            )
            
            if train_count > 0 or val_count > 0:
                print(f"   ✅ Train: {train_count}, Val: {val_count}")
                total_train += train_count
                total_val += val_count
                processed_classes += 1
            else:
                print(f"   ⚠️  No images found")
        else:
            print(f"⚠️  Class directory not found: {class_name}")
    
    print()
    print("📊 Split Summary:")
    print(f"   🏷️  Classes processed: {processed_classes}/{len(tomato_classes)}")
    print(f"   🏋️  Train images: {total_train}")
    print(f"   ✅ Val images: {total_val}")
    print(f"   📊 Total images: {total_train + total_val}")
    
    if processed_classes > 0:
        print()
        print("🎉 Dataset split completed!")
        print(f"📁 Ready for training with: {args.output}")
        print()
        print("Next steps:")
        print(f"1. Update DATASET_PATH in densenet_trainer.py to: {args.output}")
        print("2. Run: python backend/models/densenet_trainer.py")
    else:
        print("❌ No tomato classes were processed. Check your source directory structure.")
    
    # Show final structure
    if processed_classes > 0:
        print()
        print("📂 Final directory structure:")
        print(f"{args.output}/")
        print("├── train/")
        print("│   ├── Tomato_Bacterial_spot/")
        print("│   ├── Tomato_Early_blight/") 
        print("│   ├── ... (other classes)")
        print("└── val/")
        print("    ├── Tomato_Bacterial_spot/")
        print("    ├── Tomato_Early_blight/")
        print("    └── ... (other classes)")


if __name__ == "__main__":
    main()