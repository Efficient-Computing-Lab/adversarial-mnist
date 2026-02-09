"""
Adversarial Retraining using Pre-generated PGD Dataset OR On-the-fly (Madry)

This module supports two modes:
1. Pre-generated: Generate fixed adversarial dataset from frozen model, then retrain
2. On-the-fly (Madry): Generate adversarial samples during training from current model

"""
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from tqdm import tqdm
import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from attacks.pgd import PGD


def generate_pgd_dataset(model, data_loader, device, epsilon=0.15, 
                          alpha=0.01, num_steps=40, save_path=None):
    """
    Generate a fixed PGD adversarial dataset using a frozen model.
    
    Args:
        model: The frozen model to generate attacks from (Model A)
        data_loader: Clean data loader
        device: Device to use
        epsilon: Perturbation bound
        alpha: PGD step size
        num_steps: Number of PGD steps
        save_path: Optional path to save the dataset
        
    Returns:
        TensorDataset of (adversarial_images, labels)
    """
    model.eval()  # Freeze model
    model.to(device)
    
    pgd_attack = PGD(model, epsilon=epsilon, alpha=alpha, 
                     num_steps=num_steps, random_start=True)
    
    all_adv_images = []
    all_labels = []
    
    print(f"Generating PGD adversarial dataset (epsilon={epsilon}, steps={num_steps})...")
    
    pbar = tqdm(data_loader, desc='Generating PGD samples')
    for images, labels in pbar:
        images, labels = images.to(device), labels.to(device)
        
        # Generate adversarial examples
        adv_images = pgd_attack.attack(images, labels)
        
        # Store on CPU to save GPU memory
        all_adv_images.append(adv_images.cpu())
        all_labels.append(labels.cpu())
    
    # Concatenate all batches
    all_adv_images = torch.cat(all_adv_images, dim=0)
    all_labels = torch.cat(all_labels, dim=0)
    
    print(f"Generated {len(all_labels)} adversarial samples")
    
    # Create dataset
    adv_dataset = TensorDataset(all_adv_images, all_labels)
    
    # Optionally save
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        torch.save({
            'adv_images': all_adv_images,
            'labels': all_labels,
            'epsilon': epsilon,
            'alpha': alpha,
            'num_steps': num_steps
        }, save_path)
        print(f"Adversarial dataset saved to {save_path}")
    
    return adv_dataset


def load_pgd_dataset(load_path):
    """Load a pre-generated PGD adversarial dataset."""
    data = torch.load(load_path)
    dataset = TensorDataset(data['adv_images'], data['labels'])
    print(f"Loaded adversarial dataset from {load_path}")
    print(f"  Samples: {len(dataset)}")
    print(f"  Epsilon: {data['epsilon']}, Steps: {data['num_steps']}")
    return dataset


def retrain_on_adversarial(model, adv_dataset, test_loader, device,
                            epochs=10, lr=0.0001, batch_size=64,
                            save_path='models/model_a_prime.pth'):
    """
    Retrain a model on a pre-generated adversarial dataset.
    
    This is OFFLINE retraining - the adversarial samples are fixed,
    not generated adaptively during training.
    
    Args:
        model: Model to retrain (initialized from Model A's weights)
        adv_dataset: Pre-generated adversarial TensorDataset
        test_loader: Clean test data for evaluation
        device: Device to use
        epochs: Number of retraining epochs
        lr: Learning rate (typically lower for fine-tuning)
        batch_size: Batch size
        save_path: Path to save the retrained model
        
    Returns:
        Training history
    """
    model = model.to(device)
    
    # Create data loader for adversarial dataset
    adv_loader = DataLoader(adv_dataset, batch_size=batch_size, shuffle=True)
    
    optimizer = optim.Adam(model.parameters(), lr=lr)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.5)
    loss_fn = nn.CrossEntropyLoss()
    
    history = {
        'train_loss': [],
        'train_acc': [],
        'test_acc': []
    }
    
    print(f"Retraining on {len(adv_dataset)} PRE-GENERATED adversarial samples")
    print(f"Learning rate: {lr} (fine-tuning)")
    print("-" * 50)
    
    for epoch in range(epochs):
        # Training phase
        model.train()
        total_loss = 0
        correct = 0
        total = 0
        
        pbar = tqdm(adv_loader, desc=f'Epoch {epoch+1}/{epochs}', leave=False)
        for adv_images, labels in pbar:
            adv_images, labels = adv_images.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(adv_images)
            loss = loss_fn(outputs, labels)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
            
            pbar.set_postfix({'loss': loss.item(), 'acc': 100. * correct / total})
        
        train_loss = total_loss / len(adv_loader)
        train_acc = 100. * correct / total
        
        # Evaluation on clean test data
        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for images, labels in test_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                _, predicted = outputs.max(1)
                total += labels.size(0)
                correct += predicted.eq(labels).sum().item()
        
        test_acc = 100. * correct / total
        
        scheduler.step()
        
        # Record history
        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['test_acc'].append(test_acc)
        
        print(f"Epoch {epoch+1}/{epochs} | "
              f"Loss: {train_loss:.4f} | "
              f"Train Acc (adv): {train_acc:.2f}% | "
              f"Test Acc (clean): {test_acc:.2f}%")
    
    # Save model
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    torch.save({
        'model_state_dict': model.state_dict(),
        'history': history,
        'epochs': epochs,
        'training_type': 'adversarial_retraining_pregenerated'
    }, save_path)
    print(f"\nModel A′ saved to {save_path}")
    
    return history


def retrain_on_the_fly(model, train_loader, test_loader, device,
                       epochs=10, lr=0.0001, epsilon=0.15, alpha=0.01, 
                       num_steps=7, save_path='models/model_a_prime.pth'):
    """
    Adversarial training using on-the-fly PGD (Madry et al. style).
    
    This is ADAPTIVE training - adversarial samples are generated fresh
    each batch using the CURRENT model state.
    
    Args:
        model: Model to train (initialized from Model A's weights)
        train_loader: Clean training data loader
        test_loader: Clean test data for evaluation
        device: Device to use
        epochs: Number of training epochs
        lr: Learning rate
        epsilon: Perturbation bound
        alpha: PGD step size
        num_steps: PGD steps per batch (fewer than eval for speed)
        save_path: Path to save the trained model
        
    Returns:
        Training history
    """
    model = model.to(device)
    
    optimizer = optim.Adam(model.parameters(), lr=lr)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.5)
    loss_fn = nn.CrossEntropyLoss()
    
    history = {
        'train_loss': [],
        'train_clean_acc': [],
        'train_adv_acc': [],
        'test_acc': []
    }
    
    print(f"Adversarial training ON-THE-FLY (Madry style)")
    print(f"PGD params: epsilon={epsilon}, alpha={alpha}, steps={num_steps}")
    print(f"Learning rate: {lr}")
    print("-" * 50)
    
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        correct_clean = 0
        correct_adv = 0
        total = 0
        
        pbar = tqdm(train_loader, desc=f'Epoch {epoch+1}/{epochs}', leave=False)
        for images, labels in pbar:
            images, labels = images.to(device), labels.to(device)
            
            # Generate PGD adversarial examples using CURRENT model
            model.eval()
            pgd_attack = PGD(model, epsilon=epsilon, alpha=alpha, 
                           num_steps=num_steps, random_start=True)
            adv_images = pgd_attack.attack(images, labels)
            model.train()
            
            # Train on adversarial examples
            optimizer.zero_grad()
            outputs_adv = model(adv_images)
            loss = loss_fn(outputs_adv, labels)
            loss.backward()
            optimizer.step()
            
            # Statistics
            total_loss += loss.item()
            _, predicted_adv = outputs_adv.max(1)
            correct_adv += predicted_adv.eq(labels).sum().item()
            
            # Clean accuracy for monitoring
            with torch.no_grad():
                outputs_clean = model(images)
                _, predicted_clean = outputs_clean.max(1)
                correct_clean += predicted_clean.eq(labels).sum().item()
            
            total += labels.size(0)
            
            pbar.set_postfix({
                'loss': loss.item(), 
                'clean': 100. * correct_clean / total,
                'adv': 100. * correct_adv / total
            })
        
        train_loss = total_loss / len(train_loader)
        train_clean_acc = 100. * correct_clean / total
        train_adv_acc = 100. * correct_adv / total
        
        # Evaluation on clean test data
        model.eval()
        correct = 0
        total_test = 0
        with torch.no_grad():
            for images, labels in test_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                _, predicted = outputs.max(1)
                total_test += labels.size(0)
                correct += predicted.eq(labels).sum().item()
        
        test_acc = 100. * correct / total_test
        
        scheduler.step()
        
        # Record history
        history['train_loss'].append(train_loss)
        history['train_clean_acc'].append(train_clean_acc)
        history['train_adv_acc'].append(train_adv_acc)
        history['test_acc'].append(test_acc)
        
        print(f"Epoch {epoch+1}/{epochs} | "
              f"Loss: {train_loss:.4f} | "
              f"Train Clean: {train_clean_acc:.2f}% | "
              f"Train Adv: {train_adv_acc:.2f}% | "
              f"Test Clean: {test_acc:.2f}%")
    
    # Save model
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    torch.save({
        'model_state_dict': model.state_dict(),
        'history': history,
        'epochs': epochs,
        'epsilon': epsilon,
        'alpha': alpha,
        'num_steps': num_steps,
        'training_type': 'adversarial_retraining_otf'
    }, save_path)
    print(f"\nModel A′ saved to {save_path}")
    
    return history
