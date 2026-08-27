import torch
import torch.nn as nn
from torch.nn import functional as F
import torch.optim as optim
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader, WeightedRandomSampler
from sklearn.metrics import roc_curve, auc, roc_auc_score, confusion_matrix, accuracy_score, classification_report
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import os
import time
import numpy as np
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
from torchvision.models import MobileNet_V3_Large_Weights
from PIL import Image


# 🚀 CONFIGURATION
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
BATCH_SIZE = 42     # Smaller batch size for better gradient estimates
EPOCHS = 200        # More epochs with early stopping
LEARNING_RATE = 1e-4  # Lower learning rate for more stable training
PATIENCE = 25       # Increased patience for better convergence
WEIGHT_DECAY = 1e-5  # Reduced weight decay to allow more flexibility
DROPOUT_RATE = 0.3  # Adjusted dropout rate
EXPERIMENT_LOG = "experiment_log.csv"
MODEL_WEIGHTS_PATH = "mobilenetv3_best_weights7.pth"

# 📊 ADVANCED DATA AUGMENTATION
train_transform = transforms.Compose([
    transforms.RandomResizedCrop(224, scale=(0.8, 1.0)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomVerticalFlip(),
    transforms.RandomRotation(15),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
    transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    transforms.RandomErasing(p=0.2, scale=(0.02, 0.2)),
])

val_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# Dataset loading with better transforms
train_dataset = datasets.ImageFolder(r"C:\Users\Asus TUF -PC\LeafSense AI training\LeafSenseProcessed\train", transform=train_transform)
val_dataset = datasets.ImageFolder(r"C:\Users\Asus TUF -PC\LeafSense AI training\LeafSenseProcessed\val", transform=val_transform)

# Implement class balancing with weighted sampling
class_counts = [0] * len(train_dataset.classes)
for _, class_idx in train_dataset.samples:
    class_counts[class_idx] += 1

# Calculate weights for each sample based on class frequency
weights = [1.0 / class_counts[class_idx] for _, class_idx in train_dataset.samples]
sampler = WeightedRandomSampler(weights, num_samples=len(weights), replacement=True)

# Use the weighted sampler for the training loader
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, sampler=sampler, num_workers=4, pin_memory=True)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4, pin_memory=True)


# 🛠️ IMPROVED MODEL ARCHITECTURE

class LeafClassifier(nn.Module):
    def __init__(self, num_classes, DROPOUT_RATE=0.25):
        super(LeafClassifier, self).__init__()
        # Use the latest weights for better initialization
        self.model = models.mobilenet_v3_large(weights=MobileNet_V3_Large_Weights.DEFAULT)
        
        # Freeze fewer layers - only freeze the first 50% of layers
        # This allows for better fine-tuning to your specific leaf dataset
        total_params = len(list(self.model.parameters()))
        for param in list(self.model.parameters())[:total_params//2]:
            param.requires_grad = False
            
        # Replace with a simpler, more robust classifier
        in_features = self.model.classifier[0].in_features
        self.model.classifier = nn.Sequential(
            nn.Linear(in_features, 512),
            nn.BatchNorm1d(512),  # Add batch normalization for stability
            nn.ReLU(inplace=True),
            nn.Dropout(DROPOUT_RATE),
            
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Dropout(DROPOUT_RATE),
            
            nn.Linear(256, num_classes)
        )
        
        # Apply better weight initialization
        self._initialize_weights()

    def forward(self, x):
        return self.model(x)
    
    def _initialize_weights(self):
        """Apply improved weight initialization to newly added layers"""
        for m in self.model.classifier.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
                
class FocalLoss(nn.Module):
    def __init__(self, alpha=1, gamma=2, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
        
    def forward(self, inputs, targets):
        CE_loss = F.cross_entropy(inputs, targets, reduction='none')
        pt = torch.exp(-CE_loss)
        F_loss = self.alpha * (1-pt)**self.gamma * CE_loss
        
        if self.reduction == 'mean':
            return torch.mean(F_loss)
        else:
            return F_loss

# Initialize the model
num_classes = len(train_dataset.classes)
model = LeafClassifier(len(train_dataset.classes), DROPOUT_RATE = 0.25).to(DEVICE)


