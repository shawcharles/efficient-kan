# An Efficient Implementation of Kolmogorov-Arnold Network (Fork)

**Note:** This repository (`github.com/shawcharles/efficient-kan`) is a fork of an efficient implementation of Kolmogorov-Arnold Networks (KAN), adapted and utilized for the analysis presented in:

*   Shaw, C. (2025). **“Rethinking Distributional IVs: KAN-Powered D-IV-LATE & Model Choice”**  
([arXiv:2506.12765](https://arxiv.org/abs/2506.12765)). *(https://github.com/shawcharles/kan-d-iv-late)*

The original `efficient-kan` library (on which this fork is based) can be found at [https://github.com/Blealtan/efficient-kan](https://github.com/Blealtan/efficient-kan) (by @Blealtan). The original KAN implementation (`pykan`) is available [here](https://github.com/KindXiaoming/pykan).

This README combines information from the original `efficient-kan` README with added usage examples and context relevant to its application in the aforementioned papers.

---

*(Original `efficient-kan` README content starts here, slightly edited for context and integrated with new sections)*

## Overview and Efficiency Gains

This repository contains an efficient PyTorch implementation of Kolmogorov-Arnold Networks (KAN). The primary motivation is to address performance issues in the original `pykan` implementation, which arose from the need to expand intermediate variables for activation functions.

For a KAN layer with `in_features` and `out_features`, `pykan` expands the input to `(batch_size, out_features, in_features)`. This implementation reformulates the computation: it activates the input with B-spline basis functions and then linearly combines them. This significantly reduces memory costs and maps computations to straightforward matrix multiplications, benefiting both forward and backward passes.

## Key Differences from Original `pykan`

1.  **Sparsification (L1 Regularization):**
    The original KAN paper proposed L1 regularization on activations for sparsification. This is computationally intensive with the expanded tensor. This `efficient-kan` implementation uses a more standard L1 regularization on the *spline weights*, accessible via the `model.regularization_loss()` method. This approach is compatible with the efficient reformulation. While it also encourages sparsity, its impact on interpretability compared to activation-based L1 may differ.

2.  **Learnable Scale for Spline Activations:**
    Original `pykan` includes a learnable scale for each spline activation. This implementation provides an option `enable_standalone_scale_spline` (default: `True` in the `KANLinear` layer) for this feature. Setting it to `False` can improve efficiency but may alter performance.

3.  **Parameter Initialization:**
    The `base_weight` and `spline_scaler` matrices are initialized with `kaiming_uniform_` (following `nn.Linear`), which showed improved performance on MNIST in the original `efficient-kan` development.

---

## Installation

You can install this specific version of the library directly from this GitHub repository:

```bash
pip install git+https://github.com/shawcharles/efficient-kan.git
```
Ensure you have PyTorch installed in your environment.

## Dependencies
-   PyTorch (refer to `pyproject.toml` in the original `Blealtan/efficient-kan` repository for specific versioning if needed, or ensure a recent version).

## Quick Start / Basic Usage

Here's a minimal example of how to define, train, and use a KAN model from this library for a binary classification task:

```python
import torch
import torch.nn as nn
import torch.optim as optim
from efficient_kan import KAN # Ensure the library is installed

# 1. Define the KAN model
# layers_hidden should be a list like [input_dim, hidden1_dim, ..., output_dim]
model = KAN(
    layers_hidden=[10, 32, 1],  # Example: 10 inputs, 1 hidden layer (32 neurons), 1 output
    grid_size=5,                # Number of grid intervals for splines
    spline_order=3,             # Order of splines (e.g., 3 for cubic)
    # Other parameters can be set here, see "Key KAN Constructor Parameters"
)

# Dummy data for demonstration
X_train = torch.randn(100, 10)  # 100 samples, 10 features
y_train = torch.randint(0, 2, (100, 1)).float() # 100 binary labels (0 or 1)

# 2. Define optimizer and loss function
optimizer = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
# For binary classification, BCEWithLogitsLoss is used as KAN outputs raw logits
criterion = nn.BCEWithLogitsLoss() 

# 3. Training loop
num_epochs = 20  # Or a fixed number of steps
regularization_strength = 1e-4 # Multiplier for KAN's internal regularization term

for epoch in range(num_epochs):
    model.train() # Set model to training mode
    optimizer.zero_grad()
    
    output_logits = model(X_train) # Forward pass
    
    # Calculate main loss based on task (e.g., classification)
    loss_main = criterion(output_logits, y_train)
    
    # Add KAN's internal regularization loss
    # The regularization_loss() method itself has parameters:
    # regularize_activation (default 1.0) and regularize_entropy (default 1.0)
    loss_reg = model.regularization_loss() 
    
    total_loss = loss_main + regularization_strength * loss_reg
    
    total_loss.backward() # Backward pass
    optimizer.step()      # Update model parameters
    
    if (epoch + 1) % 5 == 0:
        print(f"Epoch {epoch+1}/{num_epochs}, Total Loss: {total_loss.item():.4f}, Criterion Loss: {loss_main.item():.4f}, Regularization Loss: {loss_reg.item():.4f}")

# 4. Prediction (Inference)
model.eval() # Set model to evaluation mode
X_test = torch.randn(10, 10) # Dummy test data
with torch.no_grad(): # Disable gradient calculations for inference
    test_logits = model(X_test)
    test_probabilities = torch.sigmoid(test_logits) # Convert logits to probabilities for binary classification

print("\nTest Predictions (probabilities):")
print(test_probabilities)
```

## Key `KAN` Constructor Parameters

The main `KAN` class is initialized as `KAN(layers_hidden, grid_size=5, spline_order=3, ...)` and internally uses `KANLinear` layers. Key parameters include:

-   `layers_hidden` (list of ints): Defines the number of neurons in each layer, starting with input dimension and ending with output dimension. Example: `[input_dim, hidden1_dim, output_dim]`.
-   `grid_size` (int, default: 5): The number of grid intervals for the B-spline basis functions in each `KANLinear` layer.
-   `spline_order` (int, default: 3): The order of the B-spline (e.g., 3 for cubic splines).
-   `scale_noise` (float, default: 0.1): Scale of noise used for initializing spline weights.
-   `scale_base` (float, default: 1.0): Scale for base weight initialization.
-   `scale_spline` (float, default: 1.0): Scale for spline weight initialization.
-   `base_activation` (torch.nn.Module, default: `torch.nn.SiLU`): A base activation function applied to the input before the linear transformation part of a `KANLinear` layer.
-   `grid_eps` (float, default: 0.02): Controls the mixture between a uniform grid and a data-adaptive grid during `update_grid`. `0.0` means fully data-adaptive, `1.0` means fully uniform.
-   `grid_range` (list/tuple, default: `[-1, 1]`): The initial range for the spline grids.
-   `enable_standalone_scale_spline` (bool, default: `True`, for `KANLinear`): If `True`, includes an additional learnable scalar multiplier for each spline activation, similar to a feature in the original `pykan`. Setting to `False` can improve efficiency but might alter performance.

## Regularization

This library implements L1-type regularization on the spline weights to encourage sparsity. To apply it during training, you must manually retrieve the regularization term using `model.regularization_loss()` and add it to your main loss function, scaled by a chosen strength factor:

```python
# Inside your training loop:
loss_reg = model.regularization_loss(regularize_activation=1.0, regularize_entropy=1.0)
# regularize_activation: coefficient for the L1 norm of spline weights
# regularize_entropy: coefficient for the entropy of the (normalized) L1 norms

total_loss = main_criterion_loss + your_reg_strength_multiplier * loss_reg
```
You can adjust `your_reg_strength_multiplier` to control the overall impact of this regularization. The internal `regularize_activation` and `regularize_entropy` parameters can also be tuned.

## Grid Update

The `KANLinear` layers have an `update_grid(x)` method to adapt spline grids based on input data `x`. The main `KAN` model's `forward` method has an `update_grid=False` parameter. If set to `True` (e.g., `model(X_train, update_grid=True)`), it will call `layer.update_grid(x)` for each `KANLinear` layer during that forward pass. This can be done periodically during training if desired, similar to the original `pykan`'s grid update strategy.

## Citation

This repository (`github.com/shawcharles/efficient-kan`) is a fork of an efficient KAN implementation (see original at [https://github.com/Blealtan/efficient-kan](https://github.com/Blealtan/efficient-kan) by @Blealtan). This specific version was utilized in the following research:

*   Shaw, C. (2025). Nuisance Function Estimation in the DML Framework: A Tale of Two Models. arXiv preprint. *(User to add arXiv link when available)*
*   Shaw, C. (2025). Efficient Estimation of Distributional Treatment Effects with Endogenous Treatments: A Machine Learning Approach. arXiv preprint. *(User to add arXiv link when available)*

If using this code, especially in work related to the KAN-D-IV-LATE estimator presented in the papers above, please cite the relevant paper(s) and consider acknowledging the original `efficient-kan` library and the foundational `pykan` library by Liu et al. (2024).
