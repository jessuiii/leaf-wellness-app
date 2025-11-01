"""
DenseNet169 Saliency Map Visualization
Adapted from MarkoArsenovic/DeepLearning_PlantDiseases for tomato disease detection

Generates saliency maps using naive backpropagation and guided backpropagation
to visualize which parts of the image the DenseNet model focuses on for classification.

Usage:
    python densenet_saliency.py model_path.pth dataset_path image_path disease_name [--output_dir output/]

Example:
    python densenet_saliency.py trained_models/densenet169_tomato.pth dataset/ test_image.jpg "Tomato_Early_blight" --output_dir visualizations/
"""

import os
import time
import argparse
import torch
import torchvision.models as models
import torchvision.transforms as transforms
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from collections import OrderedDict
from torchvision import datasets

# Import our visualization utilities
from visualization.torchvis_util import GradType, augment_module


def parse_arguments():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='DenseNet169 Saliency Map Visualization for Tomato Disease Detection')
    parser.add_argument('model_path', help='Path to trained DenseNet169 model (.pth file)')
    parser.add_argument('dataset_path', help='Path to dataset directory (containing train/val folders)')
    parser.add_argument('image_path', help='Path to input image for visualization')
    parser.add_argument('disease_class', help='Disease class name (e.g., "Tomato_Early_blight")')
    parser.add_argument('--output_dir', default='visualizations/saliency/', help='Output directory for visualizations')
    parser.add_argument('--arch', default='densenet169', help='Model architecture (default: densenet169)')
    parser.add_argument('--num_classes', type=int, default=10, help='Number of disease classes (default: 10)')
    
    return parser.parse_args()


def load_densenet_model(model_path, num_classes, device):
    """
    Load trained DenseNet169 model from checkpoint
    
    Args:
        model_path: Path to model checkpoint
        num_classes: Number of output classes  
        device: torch device (cuda/cpu)
        
    Returns:
        Loaded PyTorch model
    """
    # Create DenseNet169 architecture
    model = models.densenet169(pretrained=False)
    model.classifier = torch.nn.Linear(model.classifier.in_features, num_classes)
    
    # Load trained weights
    checkpoint = torch.load(model_path, map_location=device)
    
    # Handle different checkpoint formats
    if 'model_state_dict' in checkpoint:
        state_dict = checkpoint['model_state_dict']
    elif 'state_dict' in checkpoint:
        state_dict = checkpoint['state_dict']
    else:
        state_dict = checkpoint
    
    # Remove 'module.' prefix if present (from DataParallel)
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        name = k.replace('module.', '') if k.startswith('module.') else k
        new_state_dict[name] = v
    
    model.load_state_dict(new_state_dict)
    model = model.to(device)
    model.eval()
    
    return model


def load_class_labels(dataset_path):
    """
    Load class labels from dataset structure
    
    Args:
        dataset_path: Path to dataset with train/val directories
        
    Returns:
        List of class names
    """
    data_transforms = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    
    # Load from train directory to get class names
    train_dataset = datasets.ImageFolder(
        os.path.join(dataset_path, 'train'), 
        transform=data_transforms
    )
    
    return train_dataset.classes


def preprocess_image(image_path):
    """
    Preprocess image for model input
    
    Args:
        image_path: Path to input image
        
    Returns:
        tuple: (preprocessed_tensor, original_image)
    """
    # Standard ImageNet preprocessing
    preprocess = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    
    # Load and preprocess image
    image = Image.open(image_path).convert('RGB')
    image_tensor = preprocess(image)
    
    return image_tensor, image


def generate_saliency_map(model, image_tensor, target_class, device, method=GradType.GUIDED):
    """
    Generate saliency map using specified gradient method
    
    Args:
        model: Trained PyTorch model
        image_tensor: Preprocessed image tensor
        target_class: Target class index
        device: torch device
        method: Gradient calculation method (NAIVE, GUIDED, DECONV)
        
    Returns:
        numpy array: Saliency map
    """
    # Set up model for gradient computation
    vis_param_dict, reset_state, remove_handles = augment_module(model)
    vis_param_dict['method'] = method
    
    # Prepare input
    image_tensor = image_tensor.unsqueeze(0).to(device)
    image_tensor.requires_grad_(True)
    
    # Forward pass
    model.zero_grad()
    if image_tensor.grad is not None:
        image_tensor.grad.zero_()
        
    outputs = model(image_tensor)
    
    # Backward pass for target class
    target_score = outputs[0, target_class]
    target_score.backward()
    
    # Extract gradients
    gradients = image_tensor.grad.data
    
    # Convert to saliency map
    saliency = torch.abs(gradients).max(dim=1)[0].squeeze().cpu().numpy()
    
    # Clean up hooks
    remove_handles()
    
    return saliency


def save_visualization(original_image, saliency_map, output_path, method_name):
    """
    Save saliency map visualization
    
    Args:
        original_image: PIL Image
        saliency_map: numpy array
        output_path: Output file path
        method_name: Name of gradient method
    """
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    # Original image
    axes[0].imshow(original_image)
    axes[0].set_title('Original Image')
    axes[0].axis('off')
    
    # Saliency map
    axes[1].imshow(saliency_map, cmap='hot', interpolation='nearest')
    axes[1].set_title(f'Saliency Map ({method_name})')
    axes[1].axis('off')
    
    # Overlay
    axes[2].imshow(original_image)
    axes[2].imshow(saliency_map, cmap='hot', alpha=0.4, interpolation='nearest')
    axes[2].set_title('Overlay')
    axes[2].axis('off')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()


def classify_image(model, image_tensor, device):
    """
    Classify image and return prediction probabilities
    
    Args:
        model: Trained model
        image_tensor: Preprocessed image
        device: torch device
        
    Returns:
        tuple: (predicted_class_index, confidence_scores)
    """
    with torch.no_grad():
        image_tensor = image_tensor.unsqueeze(0).to(device)
        outputs = model(image_tensor)
        probabilities = torch.softmax(outputs, dim=1)
        predicted_class = torch.argmax(outputs, dim=1).item()
        
    return predicted_class, probabilities.cpu().numpy()[0]


def main():
    """Main function"""
    args = parse_arguments()
    
    # Setup
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Load model and data
    print("Loading model...")
    model = load_densenet_model(args.model_path, args.num_classes, device)
    
    print("Loading class labels...")
    class_labels = load_class_labels(args.dataset_path)
    print(f"Classes: {class_labels}")
    
    # Find target class index
    if args.disease_class not in class_labels:
        print(f"Error: '{args.disease_class}' not found in classes: {class_labels}")
        return
        
    target_class_idx = class_labels.index(args.disease_class)
    print(f"Target class: {args.disease_class} (index: {target_class_idx})")
    
    # Load and preprocess image
    print("Processing image...")
    image_tensor, original_image = preprocess_image(args.image_path)
    
    # Classify image
    predicted_class, confidence_scores = classify_image(model, image_tensor, device)
    predicted_label = class_labels[predicted_class]
    confidence = confidence_scores[predicted_class]
    
    print(f"Predicted: {predicted_label} (confidence: {confidence:.3f})")
    print(f"Target class confidence: {confidence_scores[target_class_idx]:.3f}")
    
    # Generate saliency maps for different methods
    methods = [
        (GradType.NAIVE, "Naive_Backpropagation"),
        (GradType.GUIDED, "Guided_Backpropagation")
    ]
    
    base_filename = os.path.splitext(os.path.basename(args.image_path))[0]
    
    for method, method_name in methods:
        print(f"Generating {method_name} saliency map...")
        
        start_time = time.time()
        saliency_map = generate_saliency_map(
            model, image_tensor, target_class_idx, device, method
        )
        elapsed_time = time.time() - start_time
        
        # Save visualization
        output_filename = f"{base_filename}_{args.disease_class}_{method_name}.png"
        output_path = os.path.join(args.output_dir, output_filename)
        
        save_visualization(original_image, saliency_map, output_path, method_name)
        
        print(f"Saved: {output_path} (took {elapsed_time:.2f}s)")
    
    print("\nSaliency map generation completed!")
    print(f"Results saved in: {args.output_dir}")


if __name__ == "__main__":
    main()