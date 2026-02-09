"""
Visualization utilities for adversarial examples
"""
import torch
import matplotlib.pyplot as plt
import numpy as np


def visualize_adversarial_examples(model, attack, test_loader, device, 
                                    num_examples=5, save_path=None):
    """
    Visualize adversarial examples and their perturbations.
    
    Shows: Original | Perturbation | Adversarial
    """
    model.eval()
    
    # Get a batch
    images, labels = next(iter(test_loader))
    images, labels = images[:num_examples].to(device), labels[:num_examples].to(device)
    
    # Generate adversarial examples
    adv_images = attack.attack(images, labels)
    
    # Get predictions
    with torch.no_grad():
        pred_clean = model(images).argmax(dim=1)
        pred_adv = model(adv_images).argmax(dim=1)
    
    # Calculate perturbations
    perturbations = adv_images - images
    
    # Move to CPU for plotting
    images = images.cpu().numpy()
    adv_images = adv_images.cpu().numpy()
    perturbations = perturbations.cpu().numpy()
    labels = labels.cpu().numpy()
    pred_clean = pred_clean.cpu().numpy()
    pred_adv = pred_adv.cpu().numpy()
    
    # Create figure
    fig, axes = plt.subplots(num_examples, 3, figsize=(9, 3 * num_examples))
    
    for i in range(num_examples):
        # Original
        axes[i, 0].imshow(images[i, 0], cmap='gray')
        axes[i, 0].set_title(f'Original\nTrue: {labels[i]}, Pred: {pred_clean[i]}')
        axes[i, 0].axis('off')
        
        # Perturbation (scaled for visibility)
        pert = perturbations[i, 0]
        vmax = max(abs(pert.min()), abs(pert.max()))
        axes[i, 1].imshow(pert, cmap='RdBu', vmin=-vmax, vmax=vmax)
        axes[i, 1].set_title(f'Perturbation\nL∞={np.abs(pert).max():.3f}')
        axes[i, 1].axis('off')
        
        # Adversarial
        color = 'red' if pred_adv[i] != labels[i] else 'green'
        axes[i, 2].imshow(adv_images[i, 0], cmap='gray')
        axes[i, 2].set_title(f'Adversarial\nPred: {pred_adv[i]}', color=color)
        axes[i, 2].axis('off')
    
    plt.suptitle(f'Attack: {attack}', fontsize=14)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Figure saved to {save_path}")
    
    plt.close()
    return fig


def plot_training_history(history, title='Training History', save_path=None):
    """Plot training history (loss and accuracy)."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    
    # Loss plot
    axes[0].plot(history['train_loss'], label='Train Loss')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Loss')
    axes[0].set_title('Training Loss')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # Accuracy plot - handle different history formats
    if 'train_clean_acc' in history:
        # On-the-fly adversarial training history
        axes[1].plot(history['train_clean_acc'], label='Train Clean Acc', marker='o')
        axes[1].plot(history['train_adv_acc'], label='Train Adv Acc', marker='s')
        axes[1].plot(history['test_acc'], label='Test Clean Acc', marker='^')
    elif 'test_clean_acc' in history:
        # Old adversarial training history format
        axes[1].plot(history['test_clean_acc'], label='Clean Test Acc', marker='o')
        axes[1].plot(history['test_adv_acc'], label='Robust Test Acc', marker='s')
    else:
        # Standard training history
        if 'train_acc' in history:
            axes[1].plot(history['train_acc'], label='Train Acc', marker='o')
        axes[1].plot(history['test_acc'], label='Test Acc', marker='s')
    
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Accuracy (%)')
    axes[1].set_title('Accuracy')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    plt.suptitle(title, fontsize=14)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Figure saved to {save_path}")
    
    plt.close()
    return fig


def plot_comparison_bar(results, save_path=None):
    """Create bar chart comparing standard vs robust model."""
    attacks = list(results['standard_model'].keys())
    
    # Prepare data
    std_clean = [results['standard_model'][a]['clean_accuracy'] for a in attacks]
    std_adv = [results['standard_model'][a]['adversarial_accuracy'] for a in attacks]
    rob_clean = [results['robust_model'][a]['clean_accuracy'] for a in attacks]
    rob_adv = [results['robust_model'][a]['adversarial_accuracy'] for a in attacks]
    
    x = np.arange(len(attacks))
    width = 0.2
    
    fig, ax = plt.subplots(figsize=(12, 6))
    
    bars1 = ax.bar(x - 1.5*width, std_clean, width, label='Standard (Clean)', color='steelblue', alpha=0.8)
    bars2 = ax.bar(x - 0.5*width, std_adv, width, label='Standard (Adversarial)', color='steelblue', alpha=0.4)
    bars3 = ax.bar(x + 0.5*width, rob_clean, width, label='Robust (Clean)', color='darkorange', alpha=0.8)
    bars4 = ax.bar(x + 1.5*width, rob_adv, width, label='Robust (Adversarial)', color='darkorange', alpha=0.4)
    
    ax.set_xlabel('Attack')
    ax.set_ylabel('Accuracy (%)')
    ax.set_title('Standard vs Adversarially Trained Model')
    ax.set_xticks(x)
    ax.set_xticklabels([a.split('(')[0] for a in attacks])  # Simplified attack names
    ax.legend()
    ax.set_ylim(0, 105)
    ax.grid(True, alpha=0.3, axis='y')
    
    # Add value labels
    for bars in [bars1, bars2, bars3, bars4]:
        for bar in bars:
            height = bar.get_height()
            ax.annotate(f'{height:.1f}',
                       xy=(bar.get_x() + bar.get_width() / 2, height),
                       xytext=(0, 3),
                       textcoords="offset points",
                       ha='center', va='bottom', fontsize=8)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Figure saved to {save_path}")
    
    plt.close()
    return fig
