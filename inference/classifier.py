"""
Simple CNN Classifier for MNIST
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class MNISTClassifier(nn.Module):
    """
    A simple CNN for MNIST classification.
    Architecture: Conv -> Conv -> FC -> FC
    """
    
    def __init__(self):
        super(MNISTClassifier, self).__init__()
        
        # Convolutional layers
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        
        # Pooling
        self.pool = nn.MaxPool2d(2, 2)
        
        # Fully connected layers
        # After 2 pooling layers: 28 -> 14 -> 7, so 7x7x64
        self.fc1 = nn.Linear(64 * 7 * 7, 128)
        self.fc2 = nn.Linear(128, 10)
        
        # Dropout
        #self.dropout = nn.Dropout(0.25)
    
    def forward(self, x):
        # Conv block 1
        x = self.pool(F.relu(self.conv1(x)))
        
        # Conv block 2
        x = self.pool(F.relu(self.conv2(x)))
        
        # Flatten
        x = x.view(-1, 64 * 7 * 7)
        
        # FC layers
        x = F.relu(self.fc1(x))
        #x = self.dropout(x)
        x = self.fc2(x)
        
        return x
    
    def predict(self, x):
        """Return predicted class labels"""
        with torch.no_grad():
            logits = self.forward(x)
            return torch.argmax(logits, dim=1)
    
    def predict_proba(self, x):
        """Return class probabilities"""
        with torch.no_grad():
            logits = self.forward(x)
            return F.softmax(logits, dim=1)
