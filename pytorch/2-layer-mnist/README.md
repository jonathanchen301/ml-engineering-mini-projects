# 2-Layer MLP for MNIST Classification

A minimal PyTorch implementation of a 2-layer fully connected neural network for MNIST digit classification.

## Model Architecture

- Input: 28×28 MNIST images (flattened to 784)
- Layer 1: `Linear(784, 128)` + ReLU
- Layer 2: `Linear(128, 10)` (logits for 10 digit classes)
- Loss: `CrossEntropyLoss`
- Optimizer: Adam with L2 regularization

## Dataset

- **Training**: 55,000 samples (from full MNIST train set)
- **Validation**: 5,000 samples
- **Test**: 10,000 samples (held-out for final evaluation)

## Training Details

- Batch size: 128
- Epochs: 50
- Learning rate: 0.001
- Weight decay: 0.001

## Results

After 50 epochs:
- Training loss: ~0.07
- Validation accuracy: ~97%

## Notebook

Run `notebook.ipynb` to:
1. Load and preprocess MNIST data
2. Train the model
3. Visualize training curves
4. Display misclassified examples
5. Save and reload the model checkpoint