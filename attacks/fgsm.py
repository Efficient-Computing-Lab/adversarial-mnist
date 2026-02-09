"""
Fast Gradient Sign Method (FGSM) Attack

Reference: Goodfellow et al., "Explaining and Harnessing Adversarial Examples", ICLR 2015
"""
import torch
import torch.nn as nn


class FGSM:
    """
    Fast Gradient Sign Method attack.
    
    Single-step attack that perturbs input in the direction of the gradient sign.
    x_adv = x + epsilon * sign(grad_x(loss))
    """
    
    def __init__(self, model, epsilon=0.3):
        """
        Args:
            model: The target model to attack
            epsilon: Maximum perturbation (L-infinity norm)
        """
        self.model = model
        self.epsilon = epsilon
        self.loss_fn = nn.CrossEntropyLoss()
    
    def attack(self, images, labels):
        """
        Generate adversarial examples.
        
        Args:
            images: Input images (batch_size, 1, 28, 28)
            labels: True labels
            
        Returns:
            Adversarial images
        """
        
        images = images.clone().detach().requires_grad_(True)
        
        # Forward pass
        outputs = self.model(images)
        
        # Calculate loss
        loss = self.loss_fn(outputs, labels)
        
        # Backward pass to get gradients
        self.model.zero_grad()
        loss.backward()
        
        # Get sign of gradients
        grad_sign = images.grad.sign()
        
        # Create adversarial examples
        adv_images = images + self.epsilon * grad_sign
        
        # Clamp to valid image range [0, 1]
        adv_images = torch.clamp(adv_images, 0, 1)
        
        return adv_images.detach()
    
    def __repr__(self):
        return f"FGSM(epsilon={self.epsilon})"
