"""
DenseNet169 Occlusion Experiment Visualization
Adapted from MarkoArsenovic/DeepLearning_PlantDiseases for tomato disease detection

Generates occlusion heatmaps that show which regions of an image are most important
for classification by systematically masking different parts of the image.

Usage:
    python densenet_occlusion.py model_path.pth dataset_path image_path disease_name [options]

Example:
    python densenet_occlusion.py trained_models/densenet169_tomato.pth dataset/ test_image.jpg "Tomato_Early_blight" --size 50 --stride 10
"""

import os
import time
import argparse
import math
import copy
import torch
import torchvision.models as models
import torchvision.transforms as transforms
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import seaborn as sns
from collections import OrderedDict
from torchvision import datasets


def parse_arguments():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='DenseNet169 Occlusion Experiment for Tomato Disease Detection')
    parser.add_argument('model_path', help='Path to trained DenseNet169 model (.pth file)')
    parser.add_argument('dataset_path', help='Path to dataset directory (containing train/val folders)')
    parser.add_argument('image_path', help='Path to input image for occlusion analysis')
    parser.add_argument('disease_class', help='Disease class name (e.g., "Tomato_Early_blight")')
    parser.add_argument('--output_dir', default='visualizations/occlusion/', help='Output directory for visualizations')
    parser.add_argument('--size', type=int, default=50, help='Size of occlusion square (default: 50)')
    parser.add_argument('--stride', type=int, default=10, help='Stride for sliding window (default: 10)')
    parser.add_argument('--num_classes', type=int, default=10, help='Number of disease classes (default: 10)')
    parser.add_argument('--batch_size', type=int, default=8, help='Batch size for processing (default: 8)')
    
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


def create_occlusion_masks(image, occlusion_size, stride):
    """
    Create systematically occluded versions of the input image
    
    Args:
        image: PIL Image or numpy array
        occlusion_size: Size of square occlusion window
        stride: Stride for sliding the occlusion window
        
    Returns:
        tuple: (list_of_occluded_images, output_height, output_width)
    """
    if isinstance(image, Image.Image):
        img = np.array(image)
    else:
        img = np.copy(image)
    
    height, width = img.shape[:2]
    
    # Calculate output dimensions
    output_height = int(math.ceil((height - occlusion_size) / stride + 1))
    output_width = int(math.ceil((width - occlusion_size) / stride + 1))
    
    occluded_images = []
    
    # Generate occluded images
    for h in range(output_height):
        for w in range(output_width):
            # Calculate occlusion region
            h_start = h * stride
            w_start = w * stride
            h_end = min(height, h_start + occlusion_size)
            w_end = min(width, w_start + occlusion_size)
            
            # Create occluded version
            input_image = copy.deepcopy(img)
            input_image[h_start:h_end, w_start:w_end] = 0  # Black occlusion
            
            # Convert back to PIL for preprocessing
            occluded_images.append(Image.fromarray(input_image.astype(np.uint8)))
    
    return occluded_images, output_height, output_width


def run_occlusion_experiment(model, original_image, target_class, occlusion_size, stride, batch_size, device):
    """
    Run occlusion experiment on image
    
    Args:
        model: Trained PyTorch model
        original_image: PIL Image
        target_class: Target class index
        occlusion_size: Size of occlusion window
        stride: Stride for sliding window
        batch_size: Batch size for processing
        device: torch device
        
    Returns:
        numpy array: Heatmap showing class confidence for each occlusion position
    """
    # Image preprocessing
    preprocess = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    
    # Create occluded images
    print("Creating occluded images...")
    occluded_images, output_height, output_width = create_occlusion_masks(
        original_image, occlusion_size, stride
    )
    
    total_images = len(occluded_images)
    print(f"Created {total_images} occluded images ({output_height}x{output_width})")
    
    # Preprocess all images
    print("Preprocessing images...")
    preprocessed_images = []
    for img in occluded_images:
        preprocessed_images.append(preprocess(img))
    
    # Create tensor dataset
    image_tensor = torch.stack(preprocessed_images)
    labels = torch.full((total_images,), target_class, dtype=torch.long)
    
    dataset = torch.utils.data.TensorDataset(image_tensor, labels)
    dataloader = torch.utils.data.DataLoader(
        dataset, batch_size=batch_size, shuffle=False, num_workers=0
    )
    
    # Run inference on all occluded images
    print("Running occlusion experiment...")
    confidences = []
    
    with torch.no_grad():
        for batch_idx, (images, _) in enumerate(dataloader):
            images = images.to(device)
            
            outputs = model(images)
            probabilities = torch.softmax(outputs, dim=1)
            
            # Extract confidence for target class
            target_confidences = probabilities[:, target_class].cpu().numpy()
            confidences.extend(target_confidences)
            
            if (batch_idx + 1) % 10 == 0:
                print(f"Processed {(batch_idx + 1) * batch_size}/{total_images} images")
    
    # Reshape into heatmap
    heatmap = np.array(confidences[:output_height * output_width])
    heatmap = heatmap.reshape((output_height, output_width))
    
    return heatmap


def classify_image(model, image, device):
    """
    Classify original image to get baseline prediction
    
    Args:
        model: Trained model
        image: PIL Image
        device: torch device
        
    Returns:
        tuple: (predicted_class_index, confidence_scores)
    """
    preprocess = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    
    with torch.no_grad():
        image_tensor = preprocess(image).unsqueeze(0).to(device)
        outputs = model(image_tensor)
        probabilities = torch.softmax(outputs, dim=1)
        predicted_class = torch.argmax(outputs, dim=1).item()
        
    return predicted_class, probabilities.cpu().numpy()[0]


def save_occlusion_visualization(original_image, heatmap, output_path, occlusion_size, stride, 
                                target_class, confidence, predicted_class, class_labels):
    """
    Save occlusion experiment visualization
    
    Args:
        original_image: PIL Image
        heatmap: numpy array with confidence values
        output_path: Output file path
        occlusion_size: Size of occlusion window
        stride: Stride used
        target_class: Target class index
        confidence: Original confidence
        predicted_class: Predicted class index
        class_labels: List of class names
    """
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    
    # Original image
    axes[0].imshow(original_image)
    axes[0].set_title('Original Image')
    axes[0].axis('off')
    
    # Heatmap
    im = axes[1].imshow(heatmap, cmap='YlOrRd', interpolation='bilinear')
    axes[1].set_title(f'Occlusion Heatmap\n(size={occlusion_size}, stride={stride})')
    axes[1].set_xlabel('Horizontal Position')
    axes[1].set_ylabel('Vertical Position')
    plt.colorbar(im, ax=axes[1], label='Class Confidence')
    
    # Combined view
    axes[2].imshow(original_image, alpha=0.7)
    axes[2].imshow(heatmap, cmap='YlOrRd', alpha=0.5, interpolation='bilinear')
    axes[2].set_title('Overlay')
    axes[2].axis('off')
    
    # Add text information
    info_text = f"Target: {class_labels[target_class]}\n"
    info_text += f"Predicted: {class_labels[predicted_class]}\n"
    info_text += f"Confidence: {confidence:.3f}"
    
    plt.figtext(0.02, 0.02, info_text, fontsize=10, 
                bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8))
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()


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
    
    # Load image
    print("Loading image...")
    original_image = Image.open(args.image_path).convert('RGB')
    
    # Get baseline classification
    predicted_class, confidence_scores = classify_image(model, original_image, device)
    predicted_label = class_labels[predicted_class]
    original_confidence = confidence_scores[predicted_class]
    target_confidence = confidence_scores[target_class_idx]
    
    print(f"Original prediction: {predicted_label} (confidence: {original_confidence:.3f})")
    print(f"Target class confidence: {target_confidence:.3f}")
    
    # Run occlusion experiment
    print(f"Running occlusion experiment (size={args.size}, stride={args.stride})...")
    start_time = time.time()
    
    heatmap = run_occlusion_experiment(
        model, original_image, target_class_idx, 
        args.size, args.stride, args.batch_size, device
    )
    
    elapsed_time = time.time() - start_time
    print(f"Occlusion experiment completed in {elapsed_time:.2f} seconds")
    
    # Save visualization
    base_filename = os.path.splitext(os.path.basename(args.image_path))[0]
    output_filename = f"{base_filename}_{args.disease_class}_occlusion_s{args.size}_st{args.stride}.png"
    output_path = os.path.join(args.output_dir, output_filename)
    
    save_occlusion_visualization(
        original_image, heatmap, output_path,
        args.size, args.stride, target_class_idx, 
        target_confidence, predicted_class, class_labels
    )
    
    print(f"Visualization saved: {output_path}")
    
    # Print statistics
    print(f"\nOcclusion Statistics:")
    print(f"Min confidence: {heatmap.min():.3f}")
    print(f"Max confidence: {heatmap.max():.3f}")
    print(f"Mean confidence: {heatmap.mean():.3f}")
    print(f"Std confidence: {heatmap.std():.3f}")
    
    print("\nOcclusion experiment completed!")


if __name__ == "__main__":
    main()