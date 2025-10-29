"""
Physics-Informed Neural Network for Gravitational Lens Parameter Inference

This module implements a PINN that:
1. Infers lens parameters (mass, scale radius, source position, H0)
2. Classifies dark matter model type (CDM, WDM, SIDM)
3. Enforces physical constraints via physics-informed loss
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, Tuple, Optional


class PhysicsInformedNN(nn.Module):
    """
    Physics-Informed Neural Network for lens parameter inference.
    
    Architecture:
    - Input: 64×64 grayscale image (flattened → 4096)
    - Encoder: Conv2D layers [32, 64, 128] → flatten
    - Dense layers: [1024, 512, 256]
    - Dual output heads:
        * Regression: [128] → 5 parameters [M, r_s, β_x, β_y, H0]
        * Classification: [128] → 3 classes [p_CDM, p_WDM, p_SIDM]
    
    Parameters
    ----------
    input_size : int
        Size of input images (default: 64 for 64×64 images)
    dropout_rate : float
        Dropout probability for regularization (default: 0.2)
    
    Attributes
    ----------
    encoder : nn.Sequential
        Convolutional encoder network
    dense : nn.Sequential
        Dense feature extraction layers
    param_head : nn.Sequential
        Regression head for parameter prediction
    class_head : nn.Sequential
        Classification head for DM model type
    
    Examples
    --------
    >>> model = PhysicsInformedNN(input_size=64)
    >>> images = torch.randn(32, 1, 64, 64)  # batch of 32 images
    >>> params, classes = model(images)
    >>> print(params.shape)  # (32, 5) - [M, r_s, β_x, β_y, H0]
    >>> print(classes.shape)  # (32, 3) - [p_CDM, p_WDM, p_SIDM]
    """
    
    def __init__(self, input_size: int = 64, dropout_rate: float = 0.2):
        super(PhysicsInformedNN, self).__init__()
        
        self.input_size = input_size
        self.dropout_rate = dropout_rate
        
        # Convolutional Encoder
        # Input: (batch, 1, 64, 64)
        self.encoder = nn.Sequential(
            # Conv block 1: 64×64 → 32×32
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Dropout2d(dropout_rate),
            
            # Conv block 2: 32×32 → 16×16
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Dropout2d(dropout_rate),
            
            # Conv block 3: 16×16 → 8×8
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Dropout2d(dropout_rate),
        )
        
        # Calculate flattened size after convolutions
        # After 3 pooling layers: 64 → 32 → 16 → 8
        self.encoded_size = 128 * 8 * 8  # 8192
        
        # Dense layers
        self.dense = nn.Sequential(
            nn.Linear(self.encoded_size, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            
            nn.Linear(1024, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
        )
        
        # Regression head for parameter prediction
        # Output: [M_vir, r_s, β_x, β_y, H0]
        self.param_head = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(128, 5)
        )
        
        # Classification head for DM model type
        # Output: [p_CDM, p_WDM, p_SIDM]
        self.class_head = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(128, 3)
        )
        
        # Initialize weights
        self._initialize_weights()
    
    def _initialize_weights(self):
        """Initialize network weights using He initialization."""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d) or isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass through the network.
        
        Parameters
        ----------
        x : torch.Tensor
            Input images of shape (batch, 1, H, W)
        
        Returns
        -------
        params : torch.Tensor
            Predicted parameters of shape (batch, 5)
            [M_vir, r_s, β_x, β_y, H0]
        classes : torch.Tensor
            Class logits of shape (batch, 3)
            [logit_CDM, logit_WDM, logit_SIDM]
        """
        # Encode image
        encoded = self.encoder(x)
        
        # Flatten
        flattened = encoded.view(encoded.size(0), -1)
        
        # Dense feature extraction
        features = self.dense(flattened)
        
        # Dual heads
        params = self.param_head(features)
        classes = self.class_head(features)
        
        return params, classes
    
    def predict(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Predict with post-processing and interpretable output.
        
        Parameters
        ----------
        x : torch.Tensor
            Input images of shape (batch, 1, H, W)
        
        Returns
        -------
        predictions : dict
            Dictionary containing:
            - 'params': Raw parameter predictions (batch, 5)
            - 'M_vir': Virial mass in solar masses
            - 'r_s': Scale radius in kpc
            - 'beta_x': Source x position in arcsec
            - 'beta_y': Source y position in arcsec
            - 'H0': Hubble constant in km/s/Mpc
            - 'class_probs': Softmax probabilities (batch, 3)
            - 'class_labels': Predicted class indices (batch,)
            - 'dm_type': String labels ['CDM', 'WDM', 'SIDM']
        """
        self.eval()
        with torch.no_grad():
            params, class_logits = self(x)
            
            # Apply softmax to get probabilities
            class_probs = F.softmax(class_logits, dim=1)
            
            # Get predicted class
            class_labels = torch.argmax(class_probs, dim=1)
            
            # Map to string labels
            dm_types = []
            label_map = {0: 'CDM', 1: 'WDM', 2: 'SIDM'}
            for label in class_labels.cpu().numpy():
                dm_types.append(label_map[label])
            
            return {
                'params': params,
                'M_vir': params[:, 0],
                'r_s': params[:, 1],
                'beta_x': params[:, 2],
                'beta_y': params[:, 3],
                'H0': params[:, 4],
                'class_probs': class_probs,
                'class_labels': class_labels,
                'dm_type': dm_types
            }


def physics_informed_loss(
    pred_params: torch.Tensor,
    true_params: torch.Tensor,
    pred_classes: torch.Tensor,
    true_classes: torch.Tensor,
    images: torch.Tensor,
    lambda_physics: float = 0.1,
    device: str = 'cpu'
) -> Dict[str, torch.Tensor]:
    """
    Compute physics-informed loss combining:
    1. Parameter regression loss (MSE)
    2. Classification loss (Cross-entropy)
    3. Physics residual loss (lens equation violation)
    
    The physics residual enforces the lens equation:
    θ - β - α(θ) = 0
    
    Parameters
    ----------
    pred_params : torch.Tensor
        Predicted parameters (batch, 5) [M, r_s, β_x, β_y, H0]
    true_params : torch.Tensor
        True parameters (batch, 5)
    pred_classes : torch.Tensor
        Predicted class logits (batch, 3)
    true_classes : torch.Tensor
        True class labels (batch,) as integers
    images : torch.Tensor
        Input images (batch, 1, H, W) - used for physics check
    lambda_physics : float
        Weight for physics residual term (default: 0.1)
    device : str
        Device for computation ('cpu' or 'cuda')
    
    Returns
    -------
    losses : dict
        Dictionary containing:
        - 'total': Total loss
        - 'mse_params': Parameter MSE loss
        - 'ce_class': Classification cross-entropy loss
        - 'physics_residual': Physics constraint violation
    
    Examples
    --------
    >>> pred_p = torch.randn(32, 5)
    >>> true_p = torch.randn(32, 5)
    >>> pred_c = torch.randn(32, 3)
    >>> true_c = torch.randint(0, 3, (32,))
    >>> imgs = torch.randn(32, 1, 64, 64)
    >>> losses = physics_informed_loss(pred_p, true_p, pred_c, true_c, imgs)
    """
    # 1. Parameter regression loss (MSE)
    mse_params = F.mse_loss(pred_params, true_params)
    
    # 2. Classification loss (Cross-entropy)
    ce_class = F.cross_entropy(pred_classes, true_classes)
    
    # 3. Physics residual: Enforce lens equation
    # Sample points on the image plane
    batch_size = images.size(0)
    n_sample_points = 16  # Sample 16 points per image
    
    # Generate random image plane positions (in normalized coords)
    # Map from [-1, 1] to physical coordinates
    theta_x = torch.rand(batch_size, n_sample_points, device=device) * 2 - 1  # [-1, 1]
    theta_y = torch.rand(batch_size, n_sample_points, device=device) * 2 - 1
    
    # Extract predicted source positions
    beta_x = pred_params[:, 2].unsqueeze(1)  # (batch, 1)
    beta_y = pred_params[:, 3].unsqueeze(1)
    
    # Simple point mass deflection angle (normalized)
    # α(θ) = θ_E^2 / r * θ_hat
    # For simplicity, use: α ∝ M / r
    M = pred_params[:, 0].unsqueeze(1)  # (batch, 1)
    
    # Compute radius from lens center
    r = torch.sqrt(theta_x**2 + theta_y**2 + 1e-6)  # Avoid division by zero
    
    # Deflection magnitude (simplified, proportional to M/r)
    # Normalize by typical mass scale
    M_norm = M / 1e12  # Normalize by 10^12 solar masses
    alpha_mag = M_norm / (r + 0.1)  # Add small offset for stability
    
    # Deflection components
    alpha_x = alpha_mag * theta_x / r
    alpha_y = alpha_mag * theta_y / r
    
    # Lens equation residual: |θ - β - α(θ)|²
    residual_x = theta_x - beta_x - alpha_x
    residual_y = theta_y - beta_y - alpha_y
    
    physics_residual = torch.mean(residual_x**2 + residual_y**2)
    
    # Total loss
    total_loss = mse_params + ce_class + lambda_physics * physics_residual
    
    return {
        'total': total_loss,
        'mse_params': mse_params,
        'ce_class': ce_class,
        'physics_residual': physics_residual
    }


def train_step(
    model: PhysicsInformedNN,
    images: torch.Tensor,
    true_params: torch.Tensor,
    true_classes: torch.Tensor,
    optimizer: torch.optim.Optimizer,
    lambda_physics: float = 0.1,
    device: str = 'cpu'
) -> Dict[str, float]:
    """
    Perform a single training step.
    
    Parameters
    ----------
    model : PhysicsInformedNN
        The PINN model
    images : torch.Tensor
        Batch of input images (batch, 1, H, W)
    true_params : torch.Tensor
        True parameters (batch, 5)
    true_classes : torch.Tensor
        True class labels (batch,)
    optimizer : torch.optim.Optimizer
        Optimizer for parameter updates
    lambda_physics : float
        Weight for physics loss term
    device : str
        Device for computation
    
    Returns
    -------
    losses : dict
        Dictionary of loss values for logging
    """
    model.train()
    optimizer.zero_grad()
    
    # Forward pass
    pred_params, pred_classes = model(images)
    
    # Compute loss
    losses = physics_informed_loss(
        pred_params, true_params,
        pred_classes, true_classes,
        images, lambda_physics, device
    )
    
    # Backward pass
    losses['total'].backward()
    
    # Gradient clipping for stability
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
    
    # Update weights
    optimizer.step()
    
    # Convert to float for logging
    return {k: v.item() for k, v in losses.items()}


def validate_step(
    model: PhysicsInformedNN,
    images: torch.Tensor,
    true_params: torch.Tensor,
    true_classes: torch.Tensor,
    lambda_physics: float = 0.1,
    device: str = 'cpu'
) -> Dict[str, float]:
    """
    Perform validation step without gradient computation.
    
    Parameters
    ----------
    model : PhysicsInformedNN
        The PINN model
    images : torch.Tensor
        Batch of validation images
    true_params : torch.Tensor
        True parameters
    true_classes : torch.Tensor
        True class labels
    lambda_physics : float
        Weight for physics loss term
    device : str
        Device for computation
    
    Returns
    -------
    losses : dict
        Dictionary of loss values
    """
    model.eval()
    with torch.no_grad():
        pred_params, pred_classes = model(images)
        
        losses = physics_informed_loss(
            pred_params, true_params,
            pred_classes, true_classes,
            images, lambda_physics, device
        )
    
    return {k: v.item() for k, v in losses.items()}
