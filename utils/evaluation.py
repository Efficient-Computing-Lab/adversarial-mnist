"""
Evaluation utilities for testing attacks
"""
import torch
from tqdm import tqdm


def evaluate_attack(model, attack, test_loader, device, num_samples=None):
    """
    Evaluate an attack against a model.
    
    Args:
        model: The target model
        attack: Attack instance (FGSM or PGD)
        test_loader: Test data loader
        device: Device to use
        num_samples: If provided, only evaluate on this many samples
        
    Returns:
        Dictionary with clean accuracy, adversarial accuracy, success rate
    """
    model.eval()
    
    correct_clean = 0
    correct_adv = 0
    total = 0
    successful_attacks = 0
    
    pbar = tqdm(test_loader, desc=f'Evaluating {attack}', leave=False)
    for images, labels in pbar:
        if num_samples and total >= num_samples:
            break
            
        images, labels = images.to(device), labels.to(device)
        
        # Generate adversarial examples
        adv_images = attack.attack(images, labels)
        
        with torch.no_grad():
            # Clean predictions
            outputs_clean = model(images)
            _, pred_clean = outputs_clean.max(1)
            
            # Adversarial predictions
            outputs_adv = model(adv_images)
            _, pred_adv = outputs_adv.max(1)
            
            # Count correct predictions
            correct_clean += pred_clean.eq(labels).sum().item()
            correct_adv += pred_adv.eq(labels).sum().item()
            
            # Count successful attacks (originally correct -> now incorrect)
            originally_correct = pred_clean.eq(labels)
            now_incorrect = ~pred_adv.eq(labels)
            successful_attacks += (originally_correct & now_incorrect).sum().item()
            
            total += labels.size(0)
            
        pbar.set_postfix({
            'clean': f'{100.*correct_clean/total:.1f}%',
            'adv': f'{100.*correct_adv/total:.1f}%'
        })
    
    clean_acc = 100. * correct_clean / total
    adv_acc = 100. * correct_adv / total
    
    # Attack success rate = successful attacks / originally correct
    if correct_clean > 0:
        attack_success_rate = 100. * successful_attacks / correct_clean
    else:
        attack_success_rate = 0.0
    
    return {
        'clean_accuracy': clean_acc,
        'adversarial_accuracy': adv_acc,
        'attack_success_rate': attack_success_rate,
        'samples_evaluated': total
    }


def compare_robustness(standard_model, robust_model, attacks, test_loader, 
                       device, num_samples=1000):
    """
    Compare robustness of standard vs adversarially trained model.
    
    Args:
        standard_model: Standard trained model
        robust_model: Adversarially trained model
        attacks: List of attack instances
        test_loader: Test data loader
        device: Device to use
        num_samples: Number of samples to evaluate
        
    Returns:
        Comparison results dictionary
    """
    results = {
        'standard_model': {},
        'robust_model': {}
    }
    
    print("\n" + "=" * 60)
    print("Robustness Comparison")
    print("=" * 60)
    
    for attack in attacks:
        print(f"\nAttack: {attack}")
        print("-" * 40)
        
        # Evaluate standard model
        standard_model.eval()
        std_results = evaluate_attack(
            standard_model, attack, test_loader, device, num_samples
        )
        results['standard_model'][str(attack)] = std_results
        print(f"Standard Model:")
        print(f"  Clean Acc: {std_results['clean_accuracy']:.2f}%")
        print(f"  Adv Acc:   {std_results['adversarial_accuracy']:.2f}%")
        print(f"  Attack Success: {std_results['attack_success_rate']:.2f}%")
        
        # Evaluate robust model
        robust_model.eval()
        rob_results = evaluate_attack(
            robust_model, attack, test_loader, device, num_samples
        )
        results['robust_model'][str(attack)] = rob_results
        print(f"Robust Model:")
        print(f"  Clean Acc: {rob_results['clean_accuracy']:.2f}%")
        print(f"  Adv Acc:   {rob_results['adversarial_accuracy']:.2f}%")
        print(f"  Attack Success: {rob_results['attack_success_rate']:.2f}%")
        
        # Improvement
        improvement = rob_results['adversarial_accuracy'] - std_results['adversarial_accuracy']
        print(f"  → Robustness Improvement: +{improvement:.2f}%")
    
    return results
