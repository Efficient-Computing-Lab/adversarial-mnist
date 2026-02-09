#!/usr/bin/env python3
"""
Non-Adaptive Adversarial Robustness Pipeline for MNIST

GOAL: Defend against a FIXED FGSM attacker using offline PGD-based retraining.

PIPELINE:
1. Train Model A on clean MNIST
2. Attack Model A with FGSM (observe accuracy drop)
3. Generate PGD adversarial dataset using frozen Model A
4. Fine-tune Model A′ (from A's weights) on PGD dataset
5. Evaluate Model A′ against FGSM (observe recovery)
"""
import torch
import argparse
import os
import copy

from models import MNISTClassifier
from attacks import FGSM, PGD
from training import train_standard, get_mnist_loaders, evaluate
from training.adversarial_retraining import generate_pgd_dataset, retrain_on_adversarial, retrain_on_the_fly
from utils import evaluate_attack, visualize_adversarial_examples, plot_training_history


def main(args):
    # Setup device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Create directories
    os.makedirs('results', exist_ok=True)
    os.makedirs('models', exist_ok=True)
    
    # Get data loaders
    print("\n" + "=" * 60)
    print("STEP 0: Loading MNIST dataset")
    print("=" * 60)
    train_loader, test_loader = get_mnist_loaders(
        batch_size=args.batch_size, 
        data_dir=args.data_dir
    )
    print(f"Train samples: {len(train_loader.dataset)}")
    print(f"Test samples: {len(test_loader.dataset)}")
    
    # ====================
    # Train Model A on clean data
    # ====================
    print("\n" + "=" * 60)
    print("STEP 1: Train baseline classifier (Model A)")
    print("=" * 60)
    
    model_A = MNISTClassifier()
    
    if args.load_model_a and os.path.exists(args.model_a_path):
        print(f"Loading pre-trained Model A from {args.model_a_path}")
        checkpoint = torch.load(args.model_a_path, map_location=device)
        model_A.load_state_dict(checkpoint['model_state_dict'])
        model_A.to(device)
    else:
        history_A = train_standard(
            model_A, train_loader, test_loader, device,
            epochs=args.epochs_a,
            lr=args.lr,
            save_path=args.model_a_path
        )
        plot_training_history(
            history_A, 
            title='Model A: Standard Training',
            save_path='results/model_a_training.png'
        )
    
    # Evaluate Model A on clean data
    clean_acc_A = evaluate(model_A, test_loader, device)
    print(f"\nModel A clean accuracy: {clean_acc_A:.2f}%")
    
    # ====================
    # Attack Model A with FGSM
    # ====================
    print("\n" + "=" * 60)
    print("STEP 2: Attack Model A with FGSM")
    print("=" * 60)
    
    # Create FGSM attack using Model A
    fgsm_attack_A = FGSM(model_A, epsilon=args.epsilon)
    
    # Evaluate FGSM attack on Model A
    results_A = evaluate_attack(
        model_A, fgsm_attack_A, test_loader, device,
        num_samples=args.eval_samples
    )
    
    print(f"Model A under FGSM (epsilon={args.epsilon}):")
    print(f"  Clean accuracy: {results_A['clean_accuracy']:.2f}%")
    print(f"  FGSM accuracy:  {results_A['adversarial_accuracy']:.2f}%")
    print(f"  Attack success: {results_A['attack_success_rate']:.2f}%")
    
    # Visualize adversarial examples
    visualize_adversarial_examples(
        model_A, fgsm_attack_A, test_loader, device,
        num_examples=5,
        save_path='results/fgsm_attack_on_model_a.png'
    )
    
    # ====================
    # Defense (depends on --otf flag)
    # ====================
    
    # Initialize Model A′ from Model A weights
    model_A_prime = MNISTClassifier()
    model_A_prime.load_state_dict(copy.deepcopy(model_A.state_dict()))
    
    if args.otf:
        # ON-THE-FLY (Madry style): Skip pre-generation, train adaptively
        print("\n" + "=" * 60)
        print("On-the-fly adversarial training ")
        print("=" * 60)
        print("Model A′ initialized from Model A weights")
        print("Skipping pre-generation — PGD generated otf")
        
        history_A_prime = retrain_on_the_fly(
            model=model_A_prime,
            train_loader=train_loader,
            test_loader=test_loader,
            device=device,
            epochs=args.epochs_a_prime,
            lr=args.lr_retrain,
            epsilon=args.epsilon,
            alpha=args.alpha,
            num_steps=args.pgd_steps_train,
            save_path=args.model_a_prime_path
        )
    else:
        # PRE-GENERATED: Generate fixed dataset, then retrain
        print("\n" + "=" * 60)
        print("STEP 3: Generate PGD adversarial dataset using Model A")
        print("=" * 60)
        
        # Freeze Model A and generate PGD samples
        adv_dataset = generate_pgd_dataset(
            model=model_A,
            data_loader=train_loader,
            device=device,
            epsilon=args.epsilon,
            alpha=args.alpha,
            num_steps=args.pgd_steps,
            save_path='data/pgd_adversarial_dataset.pt'
        )
        
        print(f"Generated {len(adv_dataset)} adversarial samples")
        
        print("\n" + "=" * 60)
        print("STEP 4: Offline adversarial retraining (Model A → Model A′)")
        print("=" * 60)
        print("Model A′ initialized from Model A's weights")
        
        # Retrain on adversarial samples only
        history_A_prime = retrain_on_adversarial(
            model=model_A_prime,
            adv_dataset=adv_dataset,
            test_loader=test_loader,
            device=device,
            epochs=args.epochs_a_prime,
            lr=args.lr_retrain,
            batch_size=args.batch_size,
            save_path=args.model_a_prime_path
        )
    
    plot_training_history(
        history_A_prime,
        title="Model A′: Adversarial Retraining",
        save_path='results/model_a_prime_training.png'
    )
    
    # ====================
    # Evaluate hardened Model A′
    # ====================
    print("\n" + "=" * 60)
    print("STEP 5: Evaluate hardened Model A′")
    print("=" * 60)
    
    # Clean accuracy
    clean_acc_A_prime = evaluate(model_A_prime, test_loader, device)
    
    # FGSM attack on Model A′ (using FGSM generated from A′ for fair eval)
    # But also test with FGSM from Model A (the actual threat)
    fgsm_attack_A_prime = FGSM(model_A_prime, epsilon=args.epsilon)
    
    # Evaluate with FGSM generated from A′ (white-box)
    results_A_prime_whitebox = evaluate_attack(
        model_A_prime, fgsm_attack_A_prime, test_loader, device,
        num_samples=args.eval_samples
    )
    
    # Evaluate with FGSM generated from A (the actual threat model)
    results_A_prime_transfer = evaluate_attack_transfer(
        model_A, model_A_prime, test_loader, device,
        epsilon=args.epsilon, num_samples=args.eval_samples
    )
    
    print(f"\nModel A′ Results:")
    print(f"  Clean accuracy: {clean_acc_A_prime:.2f}%")
    print(f"  FGSM (white-box on A′): {results_A_prime_whitebox['adversarial_accuracy']:.2f}%")
    print(f"  FGSM (transfer from A): {results_A_prime_transfer['adversarial_accuracy']:.2f}%")
    
    # Visualize
    visualize_adversarial_examples(
        model_A_prime, fgsm_attack_A_prime, test_loader, device,
        num_examples=5,
        save_path='results/fgsm_attack_on_model_a_prime.png'
    )
    
    # ====================
    # SUMMARY
    # ====================
    print("\n" + "=" * 60)
    print("FINAL SUMMARY")
    print("=" * 60)
    
    training_mode = "On-the-fly (Madry)" if args.otf else "Pre-generated"
    print(f"Training mode: {training_mode}")
    
    print(f"\n{'Metric':<30} {'Model A':<15} {'Model A′':<15} {'Change':<15}")
    print("-" * 75)
    
    # Clean accuracy
    clean_change = clean_acc_A_prime - clean_acc_A
    print(f"{'Clean Accuracy':<30} {clean_acc_A:<15.2f} {clean_acc_A_prime:<15.2f} {clean_change:+.2f}%")
    
    # FGSM robustness (transfer attack - the actual threat)
    fgsm_A = results_A['adversarial_accuracy']
    fgsm_A_prime = results_A_prime_transfer['adversarial_accuracy']
    fgsm_change = fgsm_A_prime - fgsm_A
    print(f"{'FGSM Robustness (from A)':<30} {fgsm_A:<15.2f} {fgsm_A_prime:<15.2f} {fgsm_change:+.2f}%")
    
    # White-box FGSM (bonus info)
    fgsm_wb = results_A_prime_whitebox['adversarial_accuracy']
    print(f"{'FGSM Robustness (white-box)':<30} {'-':<15} {fgsm_wb:<15.2f} {'N/A':<15}")
    
    print("\n" + "=" * 60)
    print("INTERPRETATION")
    print("=" * 60)
    print(f"""
Model A was vulnerable to FGSM: {fgsm_A:.1f}% accuracy under attack.

After offline PGD-based retraining:
- Model A′ recovers to {fgsm_A_prime:.1f}% against FGSM attacks from Model A.
- Improvement: +{fgsm_change:.1f}% robustness.
- Clean accuracy change: {clean_change:+.1f}%

This demonstrates successful post-deployment hardening against a fixed FGSM attacker.
""")
    
    print("Results saved to 'results/' directory")
    print("=" * 60)


def evaluate_attack_transfer(source_model, target_model, test_loader, device,
                              epsilon, num_samples=None):
    """
    Evaluate transfer attack: generate FGSM from source_model, test on target_model.
    This simulates the actual threat: attacker uses FGSM on Model A, we deploy Model A′.
    """
    from tqdm import tqdm
    
    source_model.eval()
    target_model.eval()
    
    fgsm_attack = FGSM(source_model, epsilon=epsilon)
    
    correct_clean = 0
    correct_adv = 0
    total = 0
    
    pbar = tqdm(test_loader, desc='Transfer Attack Eval', leave=False)
    for images, labels in pbar:
        if num_samples and total >= num_samples:
            break
            
        images, labels = images.to(device), labels.to(device)
        
        # Generate adversarial examples using SOURCE model (Model A)
        adv_images = fgsm_attack.attack(images, labels)
        
        with torch.no_grad():
            # Evaluate on TARGET model (Model A′)
            outputs_clean = target_model(images)
            _, pred_clean = outputs_clean.max(1)
            correct_clean += pred_clean.eq(labels).sum().item()
            
            outputs_adv = target_model(adv_images)
            _, pred_adv = outputs_adv.max(1)
            correct_adv += pred_adv.eq(labels).sum().item()
            
            total += labels.size(0)
    
    return {
        'clean_accuracy': 100. * correct_clean / total,
        'adversarial_accuracy': 100. * correct_adv / total,
        'samples_evaluated': total
    }


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Non-Adaptive Adversarial Robustness Pipeline')
    
    # Data
    parser.add_argument('--data-dir', type=str, default='./data',
                        help='Directory to store MNIST data')
    parser.add_argument('--batch-size', type=int, default=64,
                        help='Batch size')
    
    # Model A training
    parser.add_argument('--epochs-a', type=int, default=10,
                        help='Epochs for training Model A')
    parser.add_argument('--lr', type=float, default=0.001,
                        help='Learning rate for Model A')
    
    # Model A′ retraining  
    parser.add_argument('--epochs-a-prime', type=int, default=10,
                        help='Epochs for retraining Model A′')
    parser.add_argument('--lr-retrain', type=float, default=0.0001,
                        help='Learning rate for retraining (lower for fine-tuning)')
    
    # Attack parameters
    parser.add_argument('--epsilon', type=float, default=0.15,
                        help='Perturbation bound (L-infinity) for FGSM and PGD')
    parser.add_argument('--alpha', type=float, default=0.01,
                        help='Step size for PGD')
    parser.add_argument('--pgd-steps', type=int, default=40,
                        help='Number of PGD steps for generating adversarial dataset')
    parser.add_argument('--pgd-steps-train', type=int, default=7,
                        help='Number of PGD steps during on-the-fly training (fewer for speed)')
    
    # Training mode
    parser.add_argument('--otf', action='store_true',
                        help='Use on-the-fly adversarial training (Madry style) instead of pre-generated')
    
    # Evaluation
    parser.add_argument('--eval-samples', type=int, default=5000,
                        help='Number of samples for evaluation')
    
    # Model paths
    parser.add_argument('--model-a-path', type=str, default='models/model_a.pth',
                        help='Path to save/load Model A')
    parser.add_argument('--model-a-prime-path', type=str, default='models/model_a_prime.pth',
                        help='Path to save Model A′')
    parser.add_argument('--load-model-a', action='store_true',
                        help='Load pre-trained Model A')
    
    args = parser.parse_args()
    main(args)
