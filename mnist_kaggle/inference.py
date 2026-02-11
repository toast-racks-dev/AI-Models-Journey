
import torch
import torch.nn.functional as F
from torchvision import datasets, transforms
import numpy as np
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix
from model import MNISTConvNet

def load_model(model_path="mnist_cnn.pth"):
    model = MNISTConvNet()
    
    # Step 2: Load the saved parameter values
    try:
        model.load_state_dict(torch.load(model_path, weights_only=True))
    except FileNotFoundError:
        print(f"\nError: Model file '{model_path}' not found!")
        print("Please run 'python train.py' first to train the model.")
        exit(1)
    
    # Step 3: Switch to evaluation mode
    model.eval()  # Disables dropout
    
    print(f"Model loaded from '{model_path}'")
    
    return model


def get_test_loader(batch_size=1000):
    # Calculate statistics from training set to match training
    temp_dataset = datasets.MNIST('./data', train=True, download=True, transform=transforms.ToTensor())
    temp_loader = torch.utils.data.DataLoader(temp_dataset, batch_size=len(temp_dataset))
    data = next(iter(temp_loader))[0]
    mean = torch.mean(data).item()
    std = torch.std(data).item()
    
    transform = transforms.Compose([transforms.ToTensor(),transforms.Normalize((mean,), (std,))])
    
    test_dataset = datasets.MNIST(root='./data',train=False,
        download=True,
        transform=transform
    )
    
    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0
    )
    
    return test_loader


def evaluate_model(model, test_loader):
    print("\n" + "=" * 60)
    print("Evaluating Model on Full Test Set...")
    print("=" * 60)
    
    all_predictions = []
    all_labels = []
    
    # Disable gradient computation for efficiency
    with torch.no_grad():
        for images, labels in test_loader:
            outputs = model(images)
            
    all_predictions = []
    all_labels = []
    
    with torch.no_grad():
        for images, labels in test_loader:
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            
            all_predictions.extend(predicted.numpy())
            all_labels.extend(labels.numpy())
            
    # Convert to numpy arrays
    all_predictions = np.array(all_predictions)
    all_labels = np.array(all_labels)
    
    # 1. Accuracy
    accuracy = 100.0 * np.sum(all_predictions == all_labels) / len(all_labels)
    
    # 2. Precision (Weighted average to account for class imbalance if any)
    precision = 100.0 * precision_score(all_labels, all_predictions, average='weighted', zero_division=0)
    
    # 3. Recall (Weighted average)
    recall = 100.0 * recall_score(all_labels, all_predictions, average='weighted', zero_division=0)
    
    # 4. F1-Score (Weighted average)
    f1 = 100.0 * f1_score(all_labels, all_predictions, average='weighted', zero_division=0)
    
    # 5. Confusion Matrix
    conf_matrix = confusion_matrix(all_labels, all_predictions)
    
    # Print Metrics
    print(f"{'Metric':<15} {'Value':<10}")
    print("-" * 30)
    print(f"{'Accuracy':<15} {accuracy:.2f}%")
    print(f"{'Precision':<15} {precision:.2f}%")
    print(f"{'Recall':<15} {recall:.2f}%")
    print(f"{'F1-Score':<15} {f1:.2f}%")
    print("=" * 60)
    
    print("\nConfusion Matrix:")
    print("-" * 60)
    print(conf_matrix)
    print("-" * 60)
    
    # Per-Class Accuracy
    class_accuracy = conf_matrix.diagonal() / conf_matrix.sum(axis=1)
    print("\nPer-Class Accuracy:")
    for i, acc in enumerate(class_accuracy):
        print(f"Digit {i}: {acc*100:.2f}%")
    print("=" * 60)


def main():
    """Main inference function."""
    print("=" * 60)
    print("MNIST CNN Inference & Evaluation")
    print("=" * 60)
    
    # Load the trained model
    model = load_model()
    
    # Get test data
    test_loader = get_test_loader()
    
    # Run full evaluation
    evaluate_model(model, test_loader)


if __name__ == "__main__":
    main()
