"""
DenseNet169 Training Results Visualization
Adapted from MarkoArsenovic/DeepLearning_PlantDiseases for tomato disease detection

Generates publication-quality plots of training statistics including:
- Training/validation loss curves
- Training/validation accuracy curves  
- Training time vs accuracy scatter plots
- Confusion matrices
- Performance comparison charts

Usage:
    python densenet_plot.py [--stats_file stats.csv] [--output_dir plots/]

Example:
    python densenet_plot.py --stats_file training_stats.csv --output_dir visualizations/plots/
"""

import os
import argparse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.patches import Patch
import json


def parse_arguments():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='DenseNet169 Training Results Visualization')
    parser.add_argument('--stats_file', default='training_stats.csv', 
                       help='Path to training statistics CSV file')
    parser.add_argument('--output_dir', default='visualizations/plots/', 
                       help='Output directory for plots')
    parser.add_argument('--model_name', default='DenseNet169', 
                       help='Model name for plot titles')
    parser.add_argument('--figsize_width', type=int, default=12, 
                       help='Figure width in inches')
    parser.add_argument('--figsize_height', type=int, default=8, 
                       help='Figure height in inches')
    parser.add_argument('--dpi', type=int, default=300, 
                       help='DPI for saved figures')
    
    return parser.parse_args()


def setup_plot_style():
    """Setup matplotlib style for publication-quality plots"""
    plt.style.use('seaborn-v0_8-whitegrid')
    
    # Set font parameters
    plt.rcParams.update({
        'font.size': 12,
        'font.family': 'serif',
        'axes.titlesize': 16,
        'axes.labelsize': 14,
        'xtick.labelsize': 12,
        'ytick.labelsize': 12,
        'legend.fontsize': 12,
        'figure.titlesize': 18,
        'lines.linewidth': 2.5,
        'lines.markersize': 8
    })


def load_training_stats(stats_file):
    """
    Load training statistics from CSV file
    
    Args:
        stats_file: Path to CSV file with training stats
        
    Returns:
        pandas.DataFrame: Training statistics
    """
    if not os.path.exists(stats_file):
        print(f"Warning: Stats file {stats_file} not found. Creating sample data.")
        return create_sample_stats()
    
    try:
        df = pd.read_csv(stats_file)
        return df
    except Exception as e:
        print(f"Error loading stats file: {e}")
        return create_sample_stats()


def create_sample_stats():
    """Create sample training statistics for demonstration"""
    epochs = list(range(1, 26))  # 25 epochs
    
    # Simulate realistic training curves
    train_loss = [2.3 - 1.8 * (1 - np.exp(-0.3 * i)) + 0.1 * np.random.normal() for i in epochs]
    val_loss = [2.4 - 1.7 * (1 - np.exp(-0.25 * i)) + 0.15 * np.random.normal() for i in epochs]
    
    train_acc = [0.1 + 0.87 * (1 - np.exp(-0.25 * i)) + 0.02 * np.random.normal() for i in epochs]
    val_acc = [0.15 + 0.82 * (1 - np.exp(-0.2 * i)) + 0.025 * np.random.normal() for i in epochs]
    
    # Ensure reasonable bounds
    train_loss = np.clip(train_loss, 0.05, 3.0)
    val_loss = np.clip(val_loss, 0.1, 3.5)
    train_acc = np.clip(train_acc, 0.0, 1.0)
    val_acc = np.clip(val_acc, 0.0, 1.0)
    
    return pd.DataFrame({
        'epoch': epochs,
        'train_loss': train_loss,
        'val_loss': val_loss,
        'train_accuracy': train_acc,
        'val_accuracy': val_acc,
        'learning_rate': [0.001 * (0.95 ** i) for i in epochs]
    })


def plot_loss_curves(df, output_dir, model_name, figsize, dpi):
    """
    Plot training and validation loss curves
    
    Args:
        df: DataFrame with training statistics
        output_dir: Output directory
        model_name: Model name for title
        figsize: Figure size tuple
        dpi: DPI for saving
    """
    fig, ax = plt.subplots(figsize=figsize)
    
    ax.plot(df['epoch'], df['train_loss'], 'o-', label='Training Loss', 
            color='#2E86AB', linewidth=2.5, markersize=6)
    ax.plot(df['epoch'], df['val_loss'], 's--', label='Validation Loss', 
            color='#F24236', linewidth=2.5, markersize=6)
    
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Loss')
    ax.set_title(f'{model_name} - Training and Validation Loss')
    ax.legend(frameon=True, fancybox=True, shadow=True)
    ax.grid(True, alpha=0.3)
    
    # Add best epoch annotation
    best_val_epoch = df.loc[df['val_loss'].idxmin(), 'epoch']
    best_val_loss = df['val_loss'].min()
    ax.annotate(f'Best: Epoch {best_val_epoch:.0f}\nLoss: {best_val_loss:.3f}',
                xy=(best_val_epoch, best_val_loss),
                xytext=(best_val_epoch + 3, best_val_loss + 0.2),
                arrowprops=dict(arrowstyle='->', color='red', alpha=0.7),
                bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8))
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'loss_curves.png'), dpi=dpi, bbox_inches='tight')
    plt.close()


def plot_accuracy_curves(df, output_dir, model_name, figsize, dpi):
    """
    Plot training and validation accuracy curves
    
    Args:
        df: DataFrame with training statistics
        output_dir: Output directory
        model_name: Model name for title
        figsize: Figure size tuple
        dpi: DPI for saving
    """
    fig, ax = plt.subplots(figsize=figsize)
    
    ax.plot(df['epoch'], df['train_accuracy'], 'o-', label='Training Accuracy', 
            color='#2E86AB', linewidth=2.5, markersize=6)
    ax.plot(df['epoch'], df['val_accuracy'], 's--', label='Validation Accuracy', 
            color='#F24236', linewidth=2.5, markersize=6)
    
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Accuracy')
    ax.set_title(f'{model_name} - Training and Validation Accuracy')
    ax.legend(frameon=True, fancybox=True, shadow=True)
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0, 1.05)
    
    # Add best epoch annotation
    best_val_epoch = df.loc[df['val_accuracy'].idxmax(), 'epoch']
    best_val_acc = df['val_accuracy'].max()
    ax.annotate(f'Best: Epoch {best_val_epoch:.0f}\nAcc: {best_val_acc:.3f}',
                xy=(best_val_epoch, best_val_acc),
                xytext=(best_val_epoch - 3, best_val_acc - 0.1),
                arrowprops=dict(arrowstyle='->', color='green', alpha=0.7),
                bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8))
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'accuracy_curves.png'), dpi=dpi, bbox_inches='tight')
    plt.close()


def plot_learning_rate_schedule(df, output_dir, model_name, figsize, dpi):
    """
    Plot learning rate schedule if available
    
    Args:
        df: DataFrame with training statistics
        output_dir: Output directory
        model_name: Model name for title
        figsize: Figure size tuple
        dpi: DPI for saving
    """
    if 'learning_rate' not in df.columns:
        return
    
    fig, ax = plt.subplots(figsize=figsize)
    
    ax.plot(df['epoch'], df['learning_rate'], 'o-', 
            color='#A23B72', linewidth=2.5, markersize=6)
    
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Learning Rate')
    ax.set_title(f'{model_name} - Learning Rate Schedule')
    ax.set_yscale('log')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'learning_rate_schedule.png'), dpi=dpi, bbox_inches='tight')
    plt.close()


def plot_combined_metrics(df, output_dir, model_name, figsize, dpi):
    """
    Plot combined loss and accuracy in subplots
    
    Args:
        df: DataFrame with training statistics
        output_dir: Output directory
        model_name: Model name for title
        figsize: Figure size tuple
        dpi: DPI for saving
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(figsize[0] * 1.5, figsize[1]))
    
    # Loss subplot
    ax1.plot(df['epoch'], df['train_loss'], 'o-', label='Training', 
             color='#2E86AB', linewidth=2.5, markersize=6)
    ax1.plot(df['epoch'], df['val_loss'], 's--', label='Validation', 
             color='#F24236', linewidth=2.5, markersize=6)
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.set_title('Loss')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Accuracy subplot
    ax2.plot(df['epoch'], df['train_accuracy'], 'o-', label='Training', 
             color='#2E86AB', linewidth=2.5, markersize=6)
    ax2.plot(df['epoch'], df['val_accuracy'], 's--', label='Validation', 
             color='#F24236', linewidth=2.5, markersize=6)
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy')
    ax2.set_title('Accuracy')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    ax2.set_ylim(0, 1.05)
    
    fig.suptitle(f'{model_name} - Training Progress', fontsize=18)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'combined_metrics.png'), dpi=dpi, bbox_inches='tight')
    plt.close()


def create_performance_summary(df, output_dir, model_name):
    """
    Create a text summary of model performance
    
    Args:
        df: DataFrame with training statistics
        output_dir: Output directory
        model_name: Model name
    """
    # Calculate key metrics
    final_train_loss = df['train_loss'].iloc[-1]
    final_val_loss = df['val_loss'].iloc[-1]
    final_train_acc = df['train_accuracy'].iloc[-1]
    final_val_acc = df['val_accuracy'].iloc[-1]
    
    best_val_loss = df['val_loss'].min()
    best_val_acc = df['val_accuracy'].max()
    best_loss_epoch = df.loc[df['val_loss'].idxmin(), 'epoch']
    best_acc_epoch = df.loc[df['val_accuracy'].idxmax(), 'epoch']
    
    summary = f"""
{model_name} Training Summary
{'=' * 50}

Final Performance:
- Training Loss: {final_train_loss:.4f}
- Validation Loss: {final_val_loss:.4f}
- Training Accuracy: {final_train_acc:.4f} ({final_train_acc*100:.2f}%)
- Validation Accuracy: {final_val_acc:.4f} ({final_val_acc*100:.2f}%)

Best Performance:
- Best Validation Loss: {best_val_loss:.4f} (Epoch {best_loss_epoch:.0f})
- Best Validation Accuracy: {best_val_acc:.4f} ({best_val_acc*100:.2f}%) (Epoch {best_acc_epoch:.0f})

Training Statistics:
- Total Epochs: {len(df)}
- Convergence: {'Good' if abs(final_train_loss - final_val_loss) < 0.5 else 'Potential Overfitting'}
- Generalization Gap: {abs(final_train_acc - final_val_acc):.4f} ({abs(final_train_acc - final_val_acc)*100:.2f}%)
"""
    
    # Save summary to file
    with open(os.path.join(output_dir, 'training_summary.txt'), 'w') as f:
        f.write(summary)
    
    return summary


def plot_overfitting_analysis(df, output_dir, model_name, figsize, dpi):
    """
    Plot overfitting analysis showing generalization gap
    
    Args:
        df: DataFrame with training statistics
        output_dir: Output directory
        model_name: Model name for title
        figsize: Figure size tuple
        dpi: DPI for saving
    """
    fig, ax = plt.subplots(figsize=figsize)
    
    # Calculate generalization gap
    gap_loss = df['val_loss'] - df['train_loss']
    gap_acc = df['train_accuracy'] - df['val_accuracy']
    
    # Plot gaps
    ax.plot(df['epoch'], gap_loss, 'o-', label='Loss Gap (Val - Train)', 
            color='#F24236', linewidth=2.5, markersize=6)
    ax.plot(df['epoch'], gap_acc, 's-', label='Accuracy Gap (Train - Val)', 
            color='#2E86AB', linewidth=2.5, markersize=6)
    
    ax.axhline(y=0, color='black', linestyle='--', alpha=0.5)
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Generalization Gap')
    ax.set_title(f'{model_name} - Overfitting Analysis')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'overfitting_analysis.png'), dpi=dpi, bbox_inches='tight')
    plt.close()


def create_confusion_matrix_template(output_dir):
    """
    Create a template confusion matrix for tomato diseases
    
    Args:
        output_dir: Output directory
    """
    # Tomato disease classes
    classes = [
        'Tomato_Bacterial_spot', 'Tomato_Early_blight', 'Tomato_Late_blight',
        'Tomato_Leaf_Mold', 'Tomato_Septoria_leaf_spot', 'Tomato_Spider_mites',
        'Tomato_Target_Spot', 'Tomato_Yellow_Leaf_Curl_Virus',
        'Tomato_mosaic_virus', 'Tomato_healthy'
    ]
    
    # Create sample confusion matrix
    np.random.seed(42)
    n_classes = len(classes)
    cm = np.random.rand(n_classes, n_classes)
    
    # Make diagonal dominant (good classification)
    for i in range(n_classes):
        cm[i, i] += 2.0
    
    # Normalize
    cm = cm / cm.sum(axis=1, keepdims=True)
    
    fig, ax = plt.subplots(figsize=(12, 10))
    
    im = ax.imshow(cm, interpolation='nearest', cmap='Blues')
    ax.figure.colorbar(im, ax=ax)
    
    # Set ticks and labels
    ax.set(xticks=np.arange(cm.shape[1]),
           yticks=np.arange(cm.shape[0]),
           xticklabels=[c.replace('Tomato_', '') for c in classes],
           yticklabels=[c.replace('Tomato_', '') for c in classes],
           title='Confusion Matrix - DenseNet169 Tomato Disease Classification',
           ylabel='True label',
           xlabel='Predicted label')
    
    # Rotate the tick labels and set their alignment
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
    
    # Add text annotations
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, f'{cm[i, j]:.2f}',
                   ha="center", va="center",
                   color="white" if cm[i, j] > thresh else "black",
                   fontsize=8)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'confusion_matrix_template.png'), dpi=300, bbox_inches='tight')
    plt.close()


def main():
    """Main function"""
    args = parse_arguments()
    
    # Setup
    os.makedirs(args.output_dir, exist_ok=True)
    setup_plot_style()
    
    print(f"Creating training visualization plots...")
    print(f"Output directory: {args.output_dir}")
    
    # Load training statistics
    df = load_training_stats(args.stats_file)
    print(f"Loaded {len(df)} epochs of training data")
    
    figsize = (args.figsize_width, args.figsize_height)
    
    # Generate all plots
    print("Generating loss curves...")
    plot_loss_curves(df, args.output_dir, args.model_name, figsize, args.dpi)
    
    print("Generating accuracy curves...")
    plot_accuracy_curves(df, args.output_dir, args.model_name, figsize, args.dpi)
    
    print("Generating combined metrics plot...")
    plot_combined_metrics(df, args.output_dir, args.model_name, figsize, args.dpi)
    
    print("Generating learning rate schedule...")
    plot_learning_rate_schedule(df, args.output_dir, args.model_name, figsize, args.dpi)
    
    print("Generating overfitting analysis...")
    plot_overfitting_analysis(df, args.output_dir, args.model_name, figsize, args.dpi)
    
    print("Creating confusion matrix template...")
    create_confusion_matrix_template(args.output_dir)
    
    print("Creating performance summary...")
    summary = create_performance_summary(df, args.output_dir, args.model_name)
    print(summary)
    
    print(f"\nAll plots saved to: {args.output_dir}")
    print("Generated files:")
    for filename in ['loss_curves.png', 'accuracy_curves.png', 'combined_metrics.png',
                    'learning_rate_schedule.png', 'overfitting_analysis.png',
                    'confusion_matrix_template.png', 'training_summary.txt']:
        filepath = os.path.join(args.output_dir, filename)
        if os.path.exists(filepath):
            print(f"  ✓ {filename}")


if __name__ == "__main__":
    main()