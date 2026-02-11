import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import time

# Import our model architecture
from model import MNISTConvNet

BATCH_SIZE = 64         
LEARNING_RATE = 0.001   
WEIGHT_DECAY = 1e-4     # L2 regularization to prevent overfitting
NUM_EPOCHS = 7         
RANDOM_SEED = 42        

def set_seed(seed):
    torch.manual_seed(seed)

def get_data_loaders(batch_size):
    # to calculate statistics
    temp_dataset = datasets.MNIST('./data', train=True, download=True, transform=transforms.ToTensor())
    temp_loader = DataLoader(temp_dataset, batch_size=len(temp_dataset))
    data = next(iter(temp_loader))[0]
    mean = torch.mean(data).item()
    std = torch.std(data).item()   
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=(mean,), std=(std,))
    ])
    
    # training dataset
    train_dataset = datasets.MNIST(root='./data',train=True,download=True,transform=transform)
    
    # test dataset
    test_dataset = datasets.MNIST(root='./data',train=False,download=True,transform=transform)
    
    train_loader = DataLoader(train_dataset,batch_size=batch_size,shuffle=True,num_workers=0) 
    test_loader = DataLoader(test_dataset,batch_size=batch_size,shuffle=False,num_workers=0)
    
    print(f"Training samples: {len(train_dataset):,}")
    print(f"Test samples: {len(test_dataset):,}")
    
    return train_loader, test_loader

def train_one_epoch(model, train_loader, criterion, optimizer):
    model.train()  # Enable training mode
    total_loss = 0.0
    num_batches = 0
    
    for images, labels in train_loader:
        # Forward pass
        outputs = model(images)
        loss = criterion(outputs, labels)
        
        # Backward pass and optimization
        optimizer.zero_grad()  # Clear gradients from previous step
        loss.backward()
        optimizer.step()       # Update parameters
        
        total_loss += loss.item()
        num_batches += 1
    
    return total_loss / num_batches


def evaluate(model, test_loader):

    model.eval()
    correct = 0
    total = 0
    
    with torch.no_grad():  # Disable gradient computation for efficiency
        for images, labels in test_loader:
            
            outputs = model(images)
            _, predicted = torch.max(outputs, dim=1)  # Get class with highest score
            
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    
    accuracy = 100.0 * correct / total
    return accuracy


def main():
    """Main training function."""
    print("=" * 60)
    print("MNIST CNN Training")
    print("=" * 60)
    
    # Set random seed for reproducibility
    set_seed(RANDOM_SEED)
    
    # Load data
    train_loader, test_loader = get_data_loaders(BATCH_SIZE)
    print("-" * 60)
    
    # Initialize model, loss function, and optimizer
    model = MNISTConvNet()
    criterion = nn.CrossEntropyLoss()  # Combines LogSoftmax and NLLLoss
    optimizer = optim.Adam(
        model.parameters(),
        lr=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY  # L2 regularization
    )
    
    print(f"Optimizer: Adam (lr={LEARNING_RATE}, weight_decay={WEIGHT_DECAY})")

    print(f"Batch size: {BATCH_SIZE}")
    print(f"Epochs: {NUM_EPOCHS}")
    print("=" * 60)
    
    # Training loop
    start_time = time.time()
    
    for epoch in range(1, NUM_EPOCHS + 1):
        epoch_start = time.time()
        
        # Train for one epoch
        train_loss = train_one_epoch(model, train_loader, criterion, optimizer)
        
        # Evaluate on test set
        test_accuracy = evaluate(model, test_loader)
        
        epoch_time = time.time() - epoch_start
        
        # Print progress
        print(f"Epoch [{epoch:2d}/{NUM_EPOCHS}] | "
              f"Train Loss: {train_loss:.4f} | "
              f"Test Accuracy: {test_accuracy:.2f}% | "
              f"Time: {epoch_time:.1f}s")
    
    total_time = time.time() - start_time
    print("=" * 60)
    print(f"Training completed in {total_time:.1f} seconds")
    print(f"Final Test Accuracy: {test_accuracy:.2f}%")
    
    # Save the trained model (state_dict only for portability)
    model_path = "mnist_cnn.pth"
    torch.save(model.state_dict(), model_path)
    print(f"Model saved to '{model_path}'")
    print("=" * 60)


if __name__ == "__main__":
    main()
