"""
Training Script for Marko Models
Based on: MarkoArsenovic/DeepLearning_PlantDiseases

This script implements the training approach from the Marko repository
with transfer learning and multiple training modes.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
import torchvision.transforms as transforms
from torchvision import datasets
import time
import os
import copy
from marko_models import MarkoPlantDiseaseClassifier, get_marko_model_info


class MarkoModelTrainer:
    """Training pipeline for Marko plant disease models"""
    
    def __init__(self, data_dir, model_type='densenet169', num_classes=39):
        """
        Initialize trainer
        
        Args:
            data_dir (str): Path to PlantVillage dataset directory
            model_type (str): Type of model to train
            num_classes (int): Number of classes (39 for PlantVillage)
        """
        self.data_dir = data_dir
        self.model_type = model_type
        self.num_classes = num_classes
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.batch_size = 20  # As used in repository
        self.epochs = 15
        
        # Input sizes from repository
        self.input_sizes = {
            'alexnet': (224, 224),
            'densenet169': (224, 224), 
            'resnet34': (224, 224),
            'inception_v3': (299, 299),
            'squeezenet1_1': (224, 224),
            'vgg13': (224, 224)
        }
        
        self.resize = self.input_sizes.get(model_type, (224, 224))
        
    def load_data(self):
        """Load and preprocess data (based on repository approach)"""
        
        data_transforms = {
            'train': transforms.Compose([
                transforms.RandomSizedCrop(max(self.resize)),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ]),
            'val': transforms.Compose([
                # Higher scale-up for inception
                transforms.Scale(int(max(self.resize)/224*256)),
                transforms.CenterCrop(max(self.resize)),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ]),
        }
        
        dsets = {
            x: datasets.ImageFolder(
                os.path.join(self.data_dir, x), 
                data_transforms[x]
            ) for x in ['train', 'val']
        }
        
        dset_loaders = {
            x: torch.utils.data.DataLoader(
                dsets[x], 
                batch_size=self.batch_size,
                shuffle=True
            ) for x in ['train', 'val']
        }
        
        dset_sizes = {x: len(dsets[x]) for x in ['train', 'val']}
        dset_classes = dsets['train'].classes
        
        return dset_loaders['train'], dset_loaders['val'], dset_sizes, dset_classes
    
    def filtered_params(self, net, param_list=None):
        """Get filtered parameters for training (from repository)"""
        def in_param_list(s):
            if param_list is None:
                return True
            for p in param_list:
                if s.endswith(p):
                    return True
            return False
        
        params = net.named_parameters() if param_list is None \
                else (p for p in net.named_parameters() if in_param_list(p[0]))
        return params
    
    def train_model(self, model, trainloader, param_list=None):
        """
        Train the model (based on repository implementation)
        
        Args:
            model: PyTorch model to train
            trainloader: Training data loader
            param_list: List of parameters to train (None for all)
            
        Returns:
            List of training losses
        """
        def in_param_list(s):
            if param_list is None:
                return True
            for p in param_list:
                if s.endswith(p):
                    return True
            return False
        
        criterion = nn.CrossEntropyLoss()
        if torch.cuda.is_available():
            criterion = criterion.cuda()
        
        params = (p for p in self.filtered_params(model, param_list))
        
        # If fine-tuning model, turn off grad for other params
        if param_list:
            for p_fixed in (p for p in model.named_parameters() if not in_param_list(p[0])):
                p_fixed[1].requires_grad = False
        
        # Optimizer as in paper (SGD with lr=0.001, momentum=0.9)
        optimizer = optim.SGD((p[1] for p in params), lr=0.001, momentum=0.9)
        
        losses = []
        model.train()
        
        for epoch in range(self.epochs):
            running_loss = 0.0
            for i, data in enumerate(trainloader, 0):
                # Get inputs
                inputs, labels = data
                if torch.cuda.is_available():
                    inputs = Variable(inputs.cuda())
                    labels = Variable(labels.cuda())
                else:
                    inputs = Variable(inputs)
                    labels = Variable(labels)
                
                # Zero parameter gradients
                optimizer.zero_grad()
                
                # Forward + backward + optimize
                outputs = model(inputs)
                
                loss = None
                # For nets that have multiple outputs such as inception
                if isinstance(outputs, tuple):
                    loss = sum((criterion(o, labels) for o in outputs))
                else:
                    loss = criterion(outputs, labels)
                
                loss.backward()
                optimizer.step()
                
                # Print statistics
                running_loss += loss.data.item()
                if i % 30 == 29:
                    avg_loss = running_loss / 30
                    losses.append(avg_loss)
                    print(f'[{epoch + 1}, {i + 1:5d}] loss: {avg_loss:.3f}')
                    running_loss = 0.0
        
        print('Finished Training')
        return losses
    
    def evaluate_model(self, model, testloader):
        """Evaluate model on test data"""
        model.eval()
        correct = 0
        total = 0
        
        with torch.no_grad():
            for i, data in enumerate(testloader, 0):
                images, labels = data
                
                if torch.cuda.is_available():
                    images = images.cuda()
                    labels = labels.cuda()
                
                outputs = model(Variable(images))
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        
        accuracy = correct / total
        print(f'Accuracy on test images: {accuracy:.4f}')
        return accuracy
    
    def train_eval(self, model, trainloader, testloader, param_list=None):
        """Complete training and evaluation pipeline"""
        print("Training..." if not param_list else "Retraining...")
        
        before = time.time()
        losses = self.train_model(model, trainloader, param_list=param_list)
        training_time = time.time() - before
        
        print("Evaluating...")
        model.eval()
        accuracy = self.evaluate_model(model, testloader)
        
        return {
            'training_time': training_time,
            'training_losses': losses,
            'accuracy': accuracy,
            'final_loss': losses[-1] if losses else float('nan')
        }
    
    def train_with_modes(self, save_dir="trained_models"):
        """Train model with different modes: shallow, deep, from_scratch"""
        
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        
        # Load data
        print(f"Loading data for {self.model_type}...")
        trainloader, testloader, dset_sizes, dset_classes = self.load_data()
        
        results = {}
        
        # 1. Shallow retraining (only final layers)
        print(f"\n{'='*50}")
        print(f"SHALLOW RETRAINING - {self.model_type.upper()}")
        print(f"{'='*50}")
        
        classifier = MarkoPlantDiseaseClassifier(self.model_type, self.num_classes)
        model_shallow = classifier.load_model()  # Pre-trained weights
        
        if torch.cuda.is_available():
            model_shallow = torch.nn.DataParallel(model_shallow).cuda()
        
        # Get final layer parameters only
        if self.model_type == 'alexnet':
            final_params = ['classifier.6.weight', 'classifier.6.bias']
        elif self.model_type == 'densenet169':
            final_params = ['classifier.weight', 'classifier.bias']
        elif self.model_type == 'resnet34':
            final_params = ['fc.weight', 'fc.bias']
        elif self.model_type == 'vgg13':
            final_params = ['classifier.6.weight', 'classifier.6.bias']
        elif self.model_type == 'squeezenet1_1':
            final_params = ['classifier.1.weight', 'classifier.1.bias']
        else:
            final_params = None
        
        shallow_stats = self.train_eval(model_shallow, trainloader, testloader, final_params)
        shallow_stats['mode'] = 'shallow'
        results['shallow'] = shallow_stats
        
        # Save model
        shallow_path = os.path.join(save_dir, f"marko_{self.model_type}_shallow.pth")
        torch.save(model_shallow.state_dict(), shallow_path)
        results['shallow']['save_path'] = shallow_path
        
        # 2. Deep retraining (all parameters)
        print(f"\n{'='*50}")
        print(f"DEEP RETRAINING - {self.model_type.upper()}")
        print(f"{'='*50}")
        
        classifier = MarkoPlantDiseaseClassifier(self.model_type, self.num_classes)
        model_deep = classifier.load_model()  # Pre-trained weights
        
        if torch.cuda.is_available():
            model_deep = torch.nn.DataParallel(model_deep).cuda()
        
        deep_stats = self.train_eval(model_deep, trainloader, testloader, None)
        deep_stats['mode'] = 'deep'
        results['deep'] = deep_stats
        
        # Save model
        deep_path = os.path.join(save_dir, f"marko_{self.model_type}_deep.pth")
        torch.save(model_deep.state_dict(), deep_path)
        results['deep']['save_path'] = deep_path
        
        # 3. From scratch (random initialization)
        print(f"\n{'='*50}")
        print(f"FROM SCRATCH - {self.model_type.upper()}")
        print(f"{'='*50}")
        
        classifier = MarkoPlantDiseaseClassifier(self.model_type, self.num_classes)
        model_scratch = classifier.build_model()  # No pre-trained weights
        model_scratch.to(self.device)
        
        if torch.cuda.is_available():
            model_scratch = torch.nn.DataParallel(model_scratch).cuda()
        
        scratch_stats = self.train_eval(model_scratch, trainloader, testloader, None)
        scratch_stats['mode'] = 'from_scratch'
        results['from_scratch'] = scratch_stats
        
        # Save model
        scratch_path = os.path.join(save_dir, f"marko_{self.model_type}_scratch.pth")
        torch.save(model_scratch.state_dict(), scratch_path)
        results['from_scratch']['save_path'] = scratch_path
        
        return results


def train_all_models(data_dir, save_dir="trained_models"):
    """Train all Marko models with all training modes"""
    
    models_to_test = ['alexnet', 'densenet169', 'inception_v3', 'resnet34', 'squeezenet1_1', 'vgg13']
    
    if not os.path.exists(data_dir):
        print(f"Dataset directory not found: {data_dir}")
        print("Please update the data_dir path to point to your PlantVillage dataset")
        return
    
    all_results = {}
    
    for model_name in models_to_test:
        print(f"\n{'='*80}")
        print(f"TRAINING MODEL: {model_name.upper()}")
        print(f"{'='*80}")
        
        try:
            trainer = MarkoModelTrainer(data_dir, model_name, num_classes=39)
            results = trainer.train_with_modes(save_dir)
            all_results[model_name] = results
            
            # Print summary for this model
            print(f"\n{model_name.upper()} RESULTS:")
            for mode, stats in results.items():
                print(f"  {mode.upper()}: Accuracy={stats['accuracy']:.4f}, Time={stats['training_time']:.2f}s")
            
        except Exception as e:
            print(f"Error training {model_name}: {str(e)}")
            all_results[model_name] = {'error': str(e)}
    
    # Print final summary
    print(f"\n{'='*80}")
    print("FINAL RESULTS SUMMARY")
    print(f"{'='*80}")
    
    for model_name, results in all_results.items():
        if 'error' in results:
            print(f"{model_name.upper()}: FAILED - {results['error']}")
        else:
            print(f"\n{model_name.upper()}:")
            for mode, stats in results.items():
                print(f"  {mode:<12}: {stats['accuracy']:.4f} accuracy, {stats['training_time']:.2f}s")
    
    return all_results


def main():
    """Main training function"""
    # Set dataset path (update this to your PlantVillage dataset location)
    data_dir = "dataset/PlantVillage"  # Update this path
    
    if not os.path.exists(data_dir):
        print(f"Dataset directory not found: {data_dir}")
        print("Please update the data_dir path to point to your PlantVillage dataset")
        print("Expected structure:")
        print("  dataset/PlantVillage/")
        print("    train/")
        print("      Apple___Apple_scab/")
        print("      Apple___Black_rot/")
        print("      ...")
        print("    val/")
        print("      Apple___Apple_scab/")
        print("      Apple___Black_rot/")
        print("      ...")
        return
    
    # Train all models
    results = train_all_models(data_dir)
    
    # Save results to CSV (as in repository)
    import csv
    with open('marko_stats.csv', 'w', newline='') as csvfile:
        fieldnames = ['model', 'mode', 'accuracy', 'training_time', 'final_loss']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        
        writer.writeheader()
        for model_name, model_results in results.items():
            if 'error' not in model_results:
                for mode, stats in model_results.items():
                    writer.writerow({
                        'model': model_name,
                        'mode': stats['mode'],
                        'accuracy': stats['accuracy'],
                        'training_time': stats['training_time'],
                        'final_loss': stats['final_loss']
                    })
    
    print(f"\nResults saved to marko_stats.csv")


if __name__ == "__main__":
    main()