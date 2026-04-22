import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms
import time

# Import our model architecture
from model import MNISTConvNet

BATCH_SIZE = 64         
LEARNING_RATE = 0.001   
WEIGHT_DECAY = 1e-4     
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
    
    # Load training set 
    full_train_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
    
    # Split into 50k training / 10k validation
    train_size = 50000
    val_size = len(full_train_dataset) - train_size
    train_dataset, val_dataset = random_split(full_train_dataset, [train_size, val_size], generator=torch.Generator().manual_seed(RANDOM_SEED))
     
    # test dataset
    test_dataset = datasets.MNIST(root='./data', train=False, download=True, transform=transform)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True) 
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    print(f"Training samples: {len(train_dataset):,}")
    print(f"Validation samples: {len(val_dataset):,}")
    print(f"Test samples: {len(test_dataset):,}")
    
    return train_loader, val_loader, test_loader

def train_one_epoch(model, train_loader, criterion, optimizer):
    model.train()  
    total_loss = 0.0
    num_batches = 0
    
    for images, labels in train_loader:
        outputs = model(images)
        loss = criterion(outputs, labels)
        
        optimizer.zero_grad()  # Clear gradients from previous step
        loss.backward()
        optimizer.step()       
        
        total_loss += loss.item()
        num_batches += 1
    
    return total_loss / num_batches


def evaluate(model, test_loader):

    model.eval()
    correct = 0
    total = 0
    
    with torch.no_grad():  
        for images, labels in test_loader:
            
            outputs = model(images)
            _, predicted = torch.max(outputs, dim=1)
            
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    
    accuracy = 100.0 * correct / total
    return accuracy


def main():
    print("=" * 60)
    print("MNIST CNN Training")
    print("=" * 60)
    
    set_seed(RANDOM_SEED)
    
    # Load data
    train_loader, val_loader, test_loader = get_data_loaders(BATCH_SIZE)
    print("-" * 60)
    
    # Initialize model
    model = MNISTConvNet()
    criterion = nn.CrossEntropyLoss()  
    optimizer = optim.Adam(model.parameters(),lr=LEARNING_RATE,weight_decay=WEIGHT_DECAY)
    
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
        
        # Evaluate on validation set
        val_accuracy = evaluate(model, val_loader)
        
        epoch_time = time.time() - epoch_start
        
        # Print progress
        print(f"Epoch [{epoch:2d}/{NUM_EPOCHS}] | "
              f"Train Loss: {train_loss:.4f} | "
              f"Val Accuracy: {val_accuracy:.2f}% | "
              f"Time: {epoch_time:.1f}s")
    
    total_time = time.time() - start_time
    print("=" * 60)
    print("Training finished. Evaluating on hidden Test Set...")
    
    # Final evaluation on the unseen Test set
    test_accuracy = evaluate(model, test_loader)
    
    print(f"Final Test Accuracy: {test_accuracy:.2f}%")
    print(f"Total time: {total_time:.1f} seconds")
    
    # Save the trained model
    model_path = "mnist_cnn.pth"
    torch.save(model.state_dict(), model_path)
    print(f"Model saved to '{model_path}'")
    print("=" * 60)


if __name__ == "__main__":
    main()
