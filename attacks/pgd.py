"""
Projected Gradient Descent (PGD) Attack - L-infinity version

"""
import torch
import torch.nn as nn


class PGD:
    """
    Projected Gradient Descent attack with L-infinity norm constraint.
    
    Multi-step iterative attack (stronger than FGSM.)
    Each step: x_{t+1} = Proj(x_t + alpha * sign(grad_x(loss)))
    """
    
    def __init__(self, model, epsilon=0.3, alpha=0.01, num_steps=40, random_start=True):
        """
        Args:
            model: The target model to attack
            epsilon: Maximum perturbation (L-infinity bound)
            alpha: Step size for each iteration
            num_steps: Number of attack iterations
            random_start: Whether to start from a random point within epsilon ball
        """
        self.model = model
        self.epsilon = epsilon
        self.alpha = alpha
        self.num_steps = num_steps
        self.random_start = random_start
        self.loss_fn = nn.CrossEntropyLoss()
    
    def attack(self, images, labels):
        """
        Generate adversarial examples using PGD.
        
        Args:
            images: Input images (batch_size, 1, 28, 28)
            labels: True labels
            
        Returns:
            Adversarial images
        """
        # Clone original images
        original_images = images.clone().detach()
        
        # Initialize adversarial images
        if self.random_start:
            # Start from a random point within epsilon ball
            adv_images = original_images + torch.empty_like(original_images).uniform_(
                -self.epsilon, self.epsilon
            )
            adv_images = torch.clamp(adv_images, 0, 1)
        else:
            adv_images = original_images.clone()
        
        # Iterative attack
        for _ in range(self.num_steps):
            adv_images.requires_grad_(True)
            
            # Forward pass
            outputs = self.model(adv_images)
            
            # Calculate loss
            loss = self.loss_fn(outputs, labels)
            
            # Backward pass
            self.model.zero_grad()
            loss.backward()
            
            # Get gradient sign and update
            with torch.no_grad():
                grad_sign = adv_images.grad.sign()
                adv_images = adv_images + self.alpha * grad_sign
                
                # Project back onto epsilon ball (L-infinity)
                perturbation = adv_images - original_images
                perturbation = torch.clamp(perturbation, -self.epsilon, self.epsilon)
                adv_images = original_images + perturbation
                
                # Clamp to valid image range [0, 1]
                adv_images = torch.clamp(adv_images, 0, 1)
        
        return adv_images.detach()
    
    def __repr__(self):
        return f"PGD(epsilon={self.epsilon}, alpha={self.alpha}, steps={self.num_steps})"
