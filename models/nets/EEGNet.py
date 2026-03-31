"""
EEGNet model for brain-computer interface applications.

Reference:
    Lawhern, V. J., Solon, A. J., Waytowich, N. R., Gordon, S. M., Hung, C. P.,
    & Lance, B. J. (2018). EEGNet: A Compact Convolutional Neural Network for
    EEG-based Brain-Computer Interfaces. Journal of Neural Engineering, 15(5), 056013.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from config import EEGNET_TIME_STRIDE, EEGNET_POOL_SIZE


class SeparableConv2d(nn.Module):
    """
    Separable 2D Convolution = Depthwise Conv + Pointwise Conv.

    Reduces parameters while maintaining representational power.
    Common in mobile networks and resource-constrained applications.
    """

    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        stride=1,
        padding=0,
        dilation=1,
        bias=True,
    ):
        super(SeparableConv2d, self).__init__()
        self.depthwise = nn.Conv2d(
            in_channels,
            in_channels,
            kernel_size,
            stride,
            padding,
            dilation,
            groups=in_channels,
            bias=False,
        )
        self.pointwise = nn.Conv2d(in_channels, out_channels, 1, 1, 0, 1, 1, bias=bias)

    def forward(self, x):
        x = self.depthwise(x)
        x = self.pointwise(x)
        return x


def constraint(max_val, param, eps=1e-8):
    """
    Apply max-norm constraint to weights.

    Prevents weight explosion during training, especially important for
    depthwise separable convolutions.

    Parameters
    ----------
    max_val : float
        Maximum allowed norm
    param : torch.Tensor
        Parameter tensor to constrain
    eps : float
        Small value for numerical stability
    """
    norm = param.norm(2, dim=0, keepdim=True)
    desired = torch.clamp(norm, 0, max_val)
    param = param * (desired / (eps + norm))


class EEGNet(nn.Module):
    """
    EEGNet v4: Compact CNN for EEG-based Brain-Computer Interfaces.

    A lightweight neural network designed for EEG signal classification.
    Uses depthwise separable convolutions and max-norm constraints to
    minimize parameters while maintaining performance.

    Architecture:
    1. Temporal convolution (learns frequency filters)
    2. Depthwise spatial convolution (learns spatial patterns)
    3. Separable convolution for feature refinement
    4. Classification head

    Parameters
    ----------
    F1 : int
        Number of temporal convolutional filters.
        Typical values: 8, 16, 32
        - Controls frequency feature extraction
        - Larger F1 → more frequencies captured (more parameters)

    D : int
        Depth multiplier for spatial convolution.
        Typical values: 1, 2, 4
        - Total spatial filters = D * F1
        - Controls spatial feature combinations

    C : int
        Number of input channels (EEG electrodes).
        Typical values: 22-64 depending on recording setup
        - Should match your EEG headset configuration

    T : int
        Number of time samples per epoch (trial duration).
        Calculated as: T = sampling_rate_Hz * duration_seconds
        - Example: 256 Hz × 1 second = T=256
        - CONSTRAINT: T must be >= 32 (due to pooling layers)

    N : int
        Number of output classes (classification targets).
        Typical values: 2-4 for most BCI tasks
        - Binary classification: N=2
        - Multiclass (motor imagery): N=4

    dropout : float
        Dropout rate for regularization.
        Typical values: 0.25, 0.5
        - Range: 0.0-0.9
        - Prevents overfitting, especially with small datasets

    filter_len : int
        Temporal convolution kernel size.
        Default: 64
        - Rule of thumb: ~sampling_rate / 2
        - 250 Hz → 125-128
        - 256 Hz → 128

    hyper_sphere : bool
        Whether to normalize features to unit hypersphere.
        Default: False
        - If True: features projected to unit norm before classification
        - Can improve linear classifier performance

    Attributes
    ----------
    block1 : nn.Sequential
        Temporal convolution block
    block2 : nn.Sequential
        Depthwise spatial convolution + pooling
    block3 : nn.Sequential
        Separable convolution + pooling
    fc : nn.Sequential
        Classification head

    Examples
    --------
    >>> # Typical setup for 22-channel EEG at 256 Hz, 1-second trials
    >>> model = EEGNet(F1=8, D=2, C=22, T=256, N=4, dropout=0.25)
    >>>
    >>> # Input shape: (batch_size, channels, time_samples)
    >>> x = torch.randn(32, 22, 256)  # 32 trials, 22 channels, 256 samples
    >>> output = model(x)  # shape: (32, 4)
    >>>
    >>> # Get feature representations (before classification)
    >>> features = model.extract(x)  # Raw features for clustering/visualization

    Notes
    -----
    - Requires T >= 32 due to sequential pooling (4 × 8 = 32)
    - Max-norm constraints applied to depthwise and FC layer weights
    - Use with CrossEntropyLoss for classification
    - Typical parameter count: 15K-30K depending on configuration
    """

    def __init__(self, F1, D, C, T, dropout, N, filter_len=64, hyper_sphere=False):
        super(EEGNet, self).__init__()

        # Validate inputs
        if T < 32:
            raise ValueError(
                f"T must be >= 32 (due to pooling layers). "
                f"Got T={T}. Increase epoch duration or sampling rate."
            )

        # Filter configuration
        f_s = [(1, filter_len), (C, 1), (1, 4), (1, 16), (1, 8)]
        F2 = D * F1

        # Padding for temporal convolution
        l = int(np.floor((f_s[0][1] - 1) / 2))
        r = int(np.ceil((f_s[0][1] - 1) / 2))
        l2 = int(np.floor((f_s[3][1] - 1) / 2))
        r2 = int(np.ceil((f_s[3][1] - 1) / 2))

        # Block 1: Temporal Convolution
        # Learns temporal frequency patterns
        self.block1 = nn.Sequential(
            nn.ConstantPad1d((l, r), 0),
            nn.Conv2d(1, F1, f_s[0], bias=False),  # Time convolution
            nn.BatchNorm2d(F1),
        )

        # Block 2: Depthwise Spatial Convolution + Pooling
        # Learns spatial electrode relationships for each frequency
        self.block2 = nn.Sequential(
            nn.Conv2d(F1, D * F1, f_s[1], groups=F1, bias=False),  # Depthwise spatial
            nn.BatchNorm2d(D * F1),
            nn.ELU(),
            nn.AvgPool2d(f_s[2], stride=(1, 4)),  # Pool by 4
            nn.Dropout(dropout),
        )

        # Block 3: Separable Temporal Convolution + Pooling
        # Further feature refinement with fewer parameters
        self.block3 = nn.Sequential(
            nn.ConstantPad1d((l2, r2), 0),
            SeparableConv2d(D * F1, F2, f_s[3], bias=False),  # Separable temporal
            nn.BatchNorm2d(F2),
            nn.AvgPool2d(f_s[4], stride=(1, 8)),  # Pool by 8
        )

        # Classification Head
        # Linear classifier on flattened features
        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(F2 * (T // 32), N),  # T//32 = (T//4)//8
        )

        self.hyper_sphere = hyper_sphere

    def extract(self, x):
        """
        Extract features from EEG input (before classification).

        Parameters
        ----------
        x : torch.Tensor
            Input EEG data, shape (batch_size, channels, time_samples)

        Returns
        -------
        torch.Tensor
            Feature representations, shape depends on architecture
        """
        x = x.unsqueeze(1)  # Add channel dimension: (B, C, T) → (B, 1, C, T)
        self.max_norm()  # Apply max-norm constraints
        x = self.block3(self.block2(self.block1(x)))

        if self.hyper_sphere:
            # Project to unit hypersphere for easier classification
            x = F.normalize(x, p=2, dim=1)

        return x

    def forward(self, x):
        """
        Forward pass through EEGNet.

        Parameters
        ----------
        x : torch.Tensor
            Input EEG data, shape (batch_size, channels, time_samples)

        Returns
        -------
        torch.Tensor
            Classification logits, shape (batch_size, num_classes)
        """
        features = self.extract(x)
        logits = self.fc(features)
        return logits

    def max_norm(self):
        """
        Apply max-norm constraints to weights.

        Prevents weight explosion in depthwise and FC layers.
        Called during forward pass to maintain constraints.
        """
        for name, param in self.named_parameters():
            if "bias" not in name:  # Skip bias terms
                if name == "block2.0.weight":  # Depthwise conv
                    constraint(1.0, param)
                elif name == "fc.1.weight":  # FC layer
                    constraint(0.25, param)


if __name__ == "__main__":
    from torchsummary import summary

    # Example: 22-channel EEG, 256 Hz sampling, 2-second epochs (512 samples)
    model = EEGNet(F1=8, D=2, C=22, T=512, dropout=0.25, N=4, filter_len=64)

    print("Model Summary:")
    summary(model, input_size=(22, 512), device="cpu")

    print("\nTest forward pass:")
    x = torch.randn(8, 22, 512)  # 8 trials
    output = model(x)
    print(f"Output shape: {output.shape}")  # Should be (8, 4)
