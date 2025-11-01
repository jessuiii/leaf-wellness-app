"""
DenseNet169 Tomato Disease Trainer
Simplified training script focused solely on DenseNet169 for tomato disease detection

Usage:
1. Update DATASET_PATH to your tomato dataset location
2. Run: python densenet_trainer.py
3. Download trained model from 'trained_models/densenet169_tomato.pth'
"""

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from torchvision import datasets
import time
import os
import copy
from densenet_tomato_model import DenseNetTomatoClassifier


class DenseNetTrainer:
    """Simplified DenseNet169 trainer for tomato diseases"""
    
    def __init__(self, data_dir, save_dir="trained_models"):
        """Initialize DenseNet169 trainer"""
        self.data_dir = data_dir
        self.save_dir = save_dir
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Training configuration optimized for DenseNet169
        self.batch_size = 32
        self.epochs = 25
        self.learning_rate = 0.001
        self.momentum = 0.9
        
        # Create save directory
        os.makedirs(save_dir, exist_ok=True)
        
        print(f"🍅 DenseNet169 Tomato Trainer")
        print(f"📁 Dataset: {data_dir}")
        print(f"💾 Save dir: {save_dir}")
        print(f"🔧 Device: {self.device}")
        
    def load_data(self):
        """Load tomato disease dataset"""
        print("📂 Loading tomato dataset...")
        
        # Data transformations for DenseNet169
        data_transforms = {
            'train': transforms.Compose([
                transforms.Resize(256),
                transforms.RandomResizedCrop(224),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ]),
            'val': transforms.Compose([
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ]),
        }
        
        # Load datasets
        datasets_dict = {
            x: datasets.ImageFolder(
                os.path.join(self.data_dir, x), 
                data_transforms[x]
            ) for x in ['train', 'val']
        }
        
        # Data loaders
        dataloaders = {
            x: torch.utils.data.DataLoader(
                datasets_dict[x], 
                batch_size=self.batch_size,
                shuffle=(x == 'train'),
                num_workers=4
            ) for x in ['train', 'val']
        }
        
        dataset_sizes = {x: len(datasets_dict[x]) for x in ['train', 'val']}
        
        print(f"✅ Dataset loaded!")
        print(f"📊 Train: {dataset_sizes['train']}, Val: {dataset_sizes['val']}")
        
        return dataloaders, dataset_sizes
    
    def train_model(self, model, dataloaders, dataset_sizes):
        """Train DenseNet169 model"""
        print("🏋️ Training DenseNet169...")
        
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.SGD(model.parameters(), lr=self.learning_rate, momentum=self.momentum)
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)
        
        best_model_wts = copy.deepcopy(model.state_dict())
        best_acc = 0.0
        
        start_time = time.time()
        
        for epoch in range(self.epochs):
            print(f'📅 Epoch {epoch+1}/{self.epochs}')
            
            for phase in ['train', 'val']:
                if phase == 'train':
                    model.train()
                else:
                    model.eval()
                
                running_loss = 0.0
                running_corrects = 0
                
                for inputs, labels in dataloaders[phase]:
                    inputs = inputs.to(self.device)
                    labels = labels.to(self.device)
                    
                    optimizer.zero_grad()
                    
                    with torch.set_grad_enabled(phase == 'train'):
                        outputs = model(inputs)
                        _, preds = torch.max(outputs, 1)
                        loss = criterion(outputs, labels)
                        
                        if phase == 'train':
                            loss.backward()
                            optimizer.step()
                    
                    running_loss += loss.item() * inputs.size(0)
                    running_corrects += torch.sum(preds == labels.data)
                
                if phase == 'train':
                    scheduler.step()
                
                epoch_loss = running_loss / dataset_sizes[phase]
                epoch_acc = running_corrects.double() / dataset_sizes[phase]
                
                print(f'{phase.upper()} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')
                
                if phase == 'val' and epoch_acc > best_acc:
                    best_acc = epoch_acc
                    best_model_wts = copy.deepcopy(model.state_dict())
                    print(f'🎉 New best! Accuracy: {best_acc:.4f}')
        
        training_time = time.time() - start_time
        print(f'🏁 Training complete in {training_time//60:.0f}m {training_time%60:.0f}s')
        print(f'🏆 Best val accuracy: {best_acc:.4f}')
        
        model.load_state_dict(best_model_wts)
        return model, best_acc
    
    def save_model(self, model, accuracy):
        """Save trained model"""
        model_path = os.path.join(self.save_dir, 'densenet169_tomato.pth')
        torch.save(model.state_dict(), model_path)
        
        # Save training info
        info_path = os.path.join(self.save_dir, 'training_info.txt')
        with open(info_path, 'w') as f:
            f.write(f"DenseNet169 Tomato Disease Model\n")
            f.write(f"Best Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)\n")
            f.write(f"Epochs: {self.epochs}\n")
            f.write(f"Batch Size: {self.batch_size}\n")
            f.write(f"Model File: densenet169_tomato.pth\n")
        
        print(f"💾 Model saved: {model_path}")
        print(f"📝 Info saved: {info_path}")
        
        return model_path
    
    def run_training(self):
        """Execute complete training pipeline"""
        print("🚀 Starting DenseNet169 Training")
        
        if not os.path.exists(self.data_dir):
            print(f"❌ Dataset not found: {self.data_dir}")
            print("Update DATASET_PATH in the script")
            return None
        
        # Load data
        dataloaders, dataset_sizes = self.load_data()
        
        # Create model
        classifier = DenseNetTomatoClassifier()
        model = classifier.create_model()
        model = model.to(self.device)
        
        # Train
        trained_model, best_acc = self.train_model(model, dataloaders, dataset_sizes)
        
        # Save
        model_path = self.save_model(trained_model, best_acc)
        
        print(f"\n🎉 Training Complete!")
        print(f"🏆 Final accuracy: {best_acc:.4f} ({best_acc*100:.2f}%)")
        print(f"📥 Download model: {model_path}")
        
        return model_path


def main():
    """Main training function"""
    
    # 🔧 UPDATE THIS PATH to your tomato dataset
    DATASET_PATH = "dataset/tomato_disease"
    
    print("🍅 DenseNet169 Tomato Disease Trainer")
    print("="*50)
    print(f"📁 Dataset: {DATASET_PATH}")
    print()
    
    # Expected structure:
    # dataset/tomato_disease/
    #   train/
    #     Tomato___Bacterial_spot/
    #     Tomato___Early_blight/
    #     ...
    #   val/
    #     Tomato___Bacterial_spot/
    #     Tomato___Early_blight/
    #     ...
    
    trainer = DenseNetTrainer(DATASET_PATH)
    model_path = trainer.run_training()
    
    if model_path:
        print("\n" + "="*50)
        print("🍅 TRAINING COMPLETE")
        print("="*50)
        print(f"📥 Model file: {model_path}")
        print("🔧 To use:")
        print("1. Copy the .pth file")
        print("2. Load with: DenseNetTomatoClassifier().load_model(weights_path='model.pth')")


if __name__ == "__main__":
    main()