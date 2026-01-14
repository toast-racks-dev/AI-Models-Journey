# Fashion MNIST Neural Network - Single Notebook Implementation

## Core Directive
Build a production-grade neural network from scratch using **ONLY NumPy and Pandas** in a **single Jupyter notebook**. Code must be flawless, optimized, and fully functional on first execution.

---

## Mission
Classify Fashion MNIST clothing items (10 categories) with **90%+ test accuracy** using a deep neural network built entirely from scratch.

---

## Architecture Specifications

### Network Structure
```
Input Layer:     784 neurons (28×28 flattened images)
Hidden Layer 1:  256 neurons + Leaky ReLU (α=0.01) + Dropout(0.3)
Hidden Layer 2:  128 neurons + Leaky ReLU (α=0.01) + Dropout(0.3)
Hidden Layer 3:  64 neurons + Leaky ReLU (α=0.01) + Dropout(0.3)
Hidden Layer 4:  32 neurons + Leaky ReLU (α=0.01) + Dropout(0.3)
Output Layer:    10 neurons + Softmax
```

### Training Configuration
- **Optimizer**: Adam (β₁=0.9, β₂=0.999, lr=0.001, ε=1e-8)
- **Loss Function**: Categorical Cross-Entropy + L2 Regularization (λ=0.01)
- **Batch Size**: 64
- **Weight Initialization**: He Initialization
- **Early Stopping**: Patience=15 epochs on dev set accuracy
- **Max Epochs**: 150

---

## Data Pipeline

### Data Split Strategy
```
Training Data (60,000 samples from fashion-mnist-train.csv):
  ├── Train Set: 48,000 samples (80%)
  └── Dev Set:   12,000 samples (20%)

Test Data (10,000 samples from fashion-mnist-test.csv):
  └── Test Set: 10,000 samples (separate, untouched until final evaluation)
```

### Data Processing Steps
1. Load `fashion-mnist-train.csv` and `fashion-mnist-test.csv`
2. CSV format: `label, pixel1, pixel2, ..., pixel784`
3. Split training data: 80% train, 20% dev (use fixed random seed)
4. Normalize pixels: [0, 255] → [0, 1] by dividing by 255
5. One-hot encode labels (10 classes)
6. Keep test data completely separate for final evaluation

### Class Labels
```
0: T-shirt/top
1: Trouser
2: Pullover
3: Dress
4: Coat
5: Sandal
6: Shirt
7: Sneaker
8: Bag
9: Ankle boot
```

---

## Critical Implementation Details

### 1. He Initialization
```python
W = np.random.randn(n_out, n_in) * np.sqrt(2.0 / n_in)
b = np.zeros((n_out, 1))
```

### 2. Leaky ReLU Activation
```python
Forward:  f(x) = np.where(x > 0, x, 0.01 * x)
Backward: f'(x) = np.where(x > 0, 1.0, 0.01)
```

### 3. Dropout
```python
Training Mode:
  mask = (np.random.rand(*shape) > dropout_rate) / (1 - dropout_rate)
  output = input * mask

Testing Mode:
  output = input  # No dropout, no scaling needed
```

### 4. Softmax (Numerically Stable)
```python
Forward:
  z_shifted = z - np.max(z, axis=0, keepdims=True)
  exp_z = np.exp(z_shifted)
  softmax = exp_z / np.sum(exp_z, axis=0, keepdims=True)
```

### 5. Categorical Cross-Entropy Loss
```python
# Combined with softmax for numerical stability
Loss = -np.mean(np.sum(y_true * np.log(y_pred + 1e-8), axis=0))

# Gradient (when combined with softmax):
dL/dz = y_pred - y_true
```

### 6. L2 Regularization
```python
# Add to loss
L_total = L_cross_entropy + (λ / 2) * sum(np.sum(W ** 2) for all weight matrices)

# Add to gradients (DON'T regularize biases)
dW = dW_cross_entropy + λ * W
```

### 7. Adam Optimizer
```python
# Initialize
m_W = 0, v_W = 0, m_b = 0, v_b = 0, t = 0

# Update for each parameter
t = t + 1
m = β₁ * m + (1 - β₁) * gradient
v = β₂ * v + (1 - β₂) * gradient²
m_hat = m / (1 - β₁ ** t)
v_hat = v / (1 - β₂ ** t)
param = param - lr * m_hat / (np.sqrt(v_hat) + ε)
```

---

## Notebook Structure

### Section 1: Imports and Configuration
```python
import numpy as np
import pandas as pd
import time
from typing import Tuple, Dict, List

np.random.seed(42)

# Configuration dictionary with all hyperparameters
```

### Section 2: Data Loading and Preprocessing
```python
class DataLoader:
    - load_data()
    - normalize_pixels()
    - one_hot_encode()
    - train_dev_test_split()
```

### Section 3: Activation Functions
```python
class LeakyReLU:
    - forward()
    - backward()

class Softmax:
    - forward()
    - backward()  # Combined with cross-entropy
```

### Section 4: Layer Components
```python
class Dense:
    - __init__()  # He initialization
    - forward()
    - backward()

class Dropout:
    - forward(training=True)
    - backward()
```

### Section 5: Neural Network
```python
class NeuralNetwork:
    - __init__()  # Build architecture
    - forward()   # Forward pass
    - backward()  # Backward pass
    - predict()   # For inference
```

### Section 6: Optimizer
```python
class AdamOptimizer:
    - __init__()
    - update()  # Update all parameters
```

### Section 7: Loss and Metrics
```python
def categorical_cross_entropy_loss()
def l2_regularization()
def compute_accuracy()
def confusion_matrix()
def per_class_accuracy()
```

### Section 8: Training Loop
```python
class Trainer:
    - train_epoch()
    - evaluate()
    - fit()  # Main training loop with early stopping
```

### Section 9: Main Execution
```python
# 1. Load and preprocess data
# 2. Initialize network
# 3. Train with early stopping
# 4. Evaluate on test set
# 5. Display results and analysis
```

### Section 10: Results and Visualization
```python
# Display:
# - Training history (loss, train acc, dev acc per epoch)
# - Final test accuracy
# - Confusion matrix
# - Per-class accuracy
# - Most confused pairs
```

---

## Code Quality Requirements

### Must-Have Features
1. **Type hints** on all function signatures
2. **Docstrings** for all classes and methods
3. **Vectorized operations** - NO Python loops in forward/backward pass
4. **Shape assertions** at critical points
5. **NaN/Inf checks** in gradients
6. **Memory efficiency** - Use float32 dtype
7. **Progress tracking** - Print epoch stats every epoch
8. **Clean, readable code** with clear variable names

### Performance Optimizations
- Use `@` operator for matrix multiplication
- Cache activations during forward pass
- Vectorize all operations
- Use in-place operations where safe (`+=`, `*=`)
- Efficient mini-batch processing

### Key Implementation Notes
```python
# Matrix dimensions (IMPORTANT)
# Data format: (features, samples)
# X: (784, batch_size)
# W: (n_out, n_in)
# b: (n_out, 1)
# Forward: Z = W @ X + b

# Activation caching for backprop
cache = {'Z1': Z1, 'A1': A1, 'Z2': Z2, ...}

# Gradient accumulation
gradients = {'dW1': dW1, 'db1': db1, ...}
```

---

## Expected Output Format

### During Training
```
===================================
TRAINING FASHION MNIST NEURAL NETWORK
===================================
Architecture: 784 -> 256 -> 128 -> 64 -> 32 -> 10
Training samples: 48,000
Dev samples: 12,000
Test samples: 10,000

Epoch 001/150 | Loss: 0.6234 | Train Acc: 78.2% | Dev Acc: 77.5% | Time: 12.3s
Epoch 002/150 | Loss: 0.4512 | Train Acc: 83.6% | Dev Acc: 82.8% | Time: 12.1s
Epoch 003/150 | Loss: 0.3891 | Train Acc: 85.9% | Dev Acc: 85.2% | Time: 12.0s
...
Epoch 067/150 | Loss: 0.2145 | Train Acc: 92.1% | Dev Acc: 91.2% | Time: 11.9s

Early stopping triggered at epoch 67
Best dev accuracy: 91.2% (epoch 67)
```

### Final Results
```
===================================
FINAL TEST SET EVALUATION
===================================
Test Accuracy: 91.8%
Test Loss: 0.2089

Confusion Matrix:
      0    1    2    3    4    5    6    7    8    9
0  [850   0   15   20    3    0   95    0   17    0]
1  [  2 975    3    8   12    0    0    0    0    0]
2  [ 12   0  820   10   90    0   65    0    3    0]
3  [ 18   2   12  910   45    0   13    0    0    0]
4  [  1   0  102   38  858    0    1    0    0    0]
5  [  0   0    0    0    0  965    0   27    2    6]
6  [ 89   0   73   12    6    0  759    0   61    0]
7  [  0   0    0    0    0    8    0  952    1   39]
8  [  4   0    3    2    0    4   18    0  974    0]
9  [  0   0    0    0    0   12    0   43    0  955]

Per-Class Accuracy:
  T-shirt/top: 85.0%
  Trouser:     97.5%
  Pullover:    82.0%
  Dress:       91.0%
  Coat:        85.8%
  Sandal:      96.5%
  Shirt:       75.9%
  Sneaker:     95.2%
  Bag:         97.4%
  Ankle boot:  95.5%

Most Confused Pairs (Top 5):
  1. Shirt ↔ T-shirt/top: 184 mistakes
  2. Pullover ↔ Coat: 192 mistakes
  3. Coat ↔ Pullover: 102 mistakes
  4. Shirt ↔ Pullover: 138 mistakes
  5. T-shirt/top ↔ Shirt: 95 mistakes
```

---

## Common Pitfalls to Avoid

### Critical Errors
1. ❌ **Softmax overflow** - Always subtract max before exp
2. ❌ **Dropout during testing** - Must disable dropout in evaluation mode
3. ❌ **Dropout scaling** - Divide by (1 - dropout_rate) during training
4. ❌ **Adam bias correction** - Must correct for initialization bias
5. ❌ **L2 on biases** - Only regularize weights, NOT biases
6. ❌ **Data leakage** - Never touch test set until final evaluation
7. ❌ **Dimension mismatch** - Ensure (features, samples) format
8. ❌ **Gradient explosion** - Check for NaN/Inf in gradients

### Sanity Checks
```python
# Initial loss check (should be ~2.3)
initial_loss = -np.log(1/10)  # ≈ 2.302

# Overfitting test (optional but recommended)
# Train on 100 samples - should reach 100% train accuracy

# Shape validation
assert X_train.shape == (784, 48000)
assert Y_train.shape == (10, 48000)
assert X_dev.shape == (784, 12000)
assert X_test.shape == (784, 10000)
```

---

## Success Metrics

### ✅ Project Succeeds If:
- Code runs without errors on first execution
- Train/dev/test split is correct (48k/12k/10k)
- Training completes with early stopping
- Dev accuracy > 88%
- Test accuracy > 88%
- Training time < 30 minutes total
- All results are reproducible with seed=42

### 🎯 Stretch Goals:
- Test accuracy > 91%
- Training time < 20 minutes
- Implement learning rate decay
- Add training curve visualization
- Achieve 92%+ test accuracy

---

## Implementation Checklist

### Phase 1: Data (Verify First)
- [ ] Load CSV files successfully
- [ ] Split into train (48k) / dev (12k) / test (10k)
- [ ] Normalize pixels to [0, 1]
- [ ] One-hot encode labels
- [ ] Verify shapes: X=(784, n), Y=(10, n)

### Phase 2: Forward Pass
- [ ] He initialization working
- [ ] Dense layer forward pass
- [ ] Leaky ReLU activation
- [ ] Dropout (training mode)
- [ ] Softmax output
- [ ] Test with dummy data

### Phase 3: Backward Pass
- [ ] Softmax + cross-entropy gradient
- [ ] Dropout backward
- [ ] Leaky ReLU backward
- [ ] Dense layer backward
- [ ] L2 regularization gradient
- [ ] Verify gradient shapes

### Phase 4: Optimization
- [ ] Adam optimizer implementation
- [ ] Bias correction working
- [ ] Parameter updates correct
- [ ] Test on small dataset (100 samples)

### Phase 5: Training Loop
- [ ] Mini-batch processing
- [ ] Early stopping logic
- [ ] Progress printing
- [ ] Best weights saving
- [ ] Full training run

### Phase 6: Evaluation
- [ ] Accuracy computation
- [ ] Confusion matrix
- [ ] Per-class metrics
- [ ] Final test evaluation

---

## File Requirements

### Single Notebook: `fashion_mnist_nn.ipynb`

**Must include:**
- Clear section headers with markdown cells
- Inline comments for complex operations
- Print statements for debugging
- Shape assertions
- Error handling for data loading

**Data files expected:**
```
data/
└── fashion_mnist_dataset/
    ├── fashion-mnist-train.csv
    └── fashion-mnist-test.csv
```

---

## Quick Start

```python
# Cell 1: Run all cells in order
# Expected total runtime: 15-25 minutes

# Final cell output should show:
# - Test accuracy > 88%
# - Confusion matrix
# - Per-class accuracy
# - Training history
```

---

## Final Notes

This implementation demonstrates:
- ✅ Deep learning fundamentals from scratch
- ✅ NumPy vectorization and optimization
- ✅ Clean, production-ready code
- ✅ Proper train/dev/test methodology
- ✅ Regularization techniques (Dropout, L2)
- ✅ Advanced optimization (Adam)
- ✅ Early stopping and model selection

**Build this with precision. Every line matters. Achieve excellence.** 🚀