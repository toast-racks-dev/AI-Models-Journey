"""
MNIST CNN Classifier
Architecture: Input (1×28×28) → Conv(32) → Conv(64) → Conv(128) → FC(128) → Output(10)
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

class MNISTConvNet(nn.Module):    
    def __init__(self):
        super(MNISTConvNet, self).__init__()

        # Convolutional layers: 1 → 32 → 64 → 128 channels
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        
        # Fully connected layers: 1152 → 128 → 10
        # After 3 pooling operations: 28×28 → 14×14 → 7×7 → 3×3
        self.fc1 = nn.Linear(3 * 3 * 128, 128)
        self.fc2 = nn.Linear(128, 10)
        
        # Regularization
        self.dropout = nn.Dropout(0.5)
    
    def forward(self, x):
        # Conv Block 1: (batch, 1, 28, 28) → (batch, 32, 14, 14)
        x = self.pool(F.relu(self.conv1(x)))        
        
        # Conv Block 2: (batch, 32, 14, 14) → (batch, 64, 7, 7)
        x = self.pool(F.relu(self.conv2(x)))        
       
        # Conv Block 3: (batch, 64, 7, 7) → (batch, 128, 3, 3)
        x = self.pool(F.relu(self.conv3(x)))       
       
        # Flatten: (batch, 128, 3, 3) → (batch, 1152)
        x = x.view(x.size(0), -1)         
       
        # Fully connected with dropout
        x = F.relu(self.fc1(x))
        x = self.dropout(x)        
       
        # Output layer (10 classes)
        x = self.fc2(x)
        return x

