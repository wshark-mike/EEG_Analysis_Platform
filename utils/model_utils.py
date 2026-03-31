"""Deep-learning inference utilities for EEG classification.

This module provides a lightweight PyTorch-based EEGNet architecture and a
helper function for running BCI (Brain-Computer Interface) inference on
preprocessed EEG trial data.

Notes
-----
The ``SimpleEEGNet`` class is a simplified skeleton intended for demonstration
purposes.  For production use, replace it with the full network architecture
used during offline training and supply the corresponding pre-trained weights.
"""

from typing import Tuple

import torch
import torch.nn as nn
import numpy as np


class SimpleEEGNet(nn.Module):
    """A simplified EEGNet-style convolutional neural network for EEG decoding.

    Parameters
    ----------
    n_channels : int
        Number of EEG input channels.
    n_classes : int
        Number of output classes.

    Notes
    -----
    The final fully-connected layer uses ``nn.LazyLinear`` so its input
    dimension is inferred on the first forward pass.  Replace with an explicit
    ``nn.Linear`` once the temporal dimension is known.
    """

    def __init__(self, n_channels: int, n_classes: int) -> None:
        super(SimpleEEGNet, self).__init__()
        self.conv1 = nn.Conv2d(1, 16, (1, 64), padding='same')
        self.batchnorm1 = nn.BatchNorm2d(16)
        self.depthwise = nn.Conv2d(16, 32, (n_channels, 1), groups=16)
        self.flatten = nn.Flatten()
        self.fc = nn.LazyLinear(n_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Run a forward pass through the network.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor of shape ``(batch, 1, n_channels, n_times)``.

        Returns
        -------
        torch.Tensor
            Logit scores of shape ``(batch, n_classes)``.
        """
        x = self.conv1(x)
        x = self.batchnorm1(x)
        x = self.depthwise(x)
        x = nn.functional.elu(x)
        x = self.flatten(x)
        x = self.fc(x)
        return x


@torch.no_grad()
def run_bci_inference(
    data_array: np.ndarray,
    model_path: str,
    n_channels: int,
    n_classes: int = 2,
) -> Tuple[np.ndarray, np.ndarray]:
    """Run BCI model inference on preprocessed EEG trial data.

    Parameters
    ----------
    data_array : np.ndarray
        Preprocessed EEG trial data with shape
        ``(n_trials, n_channels, n_samples)``.
    model_path : str
        Path to the pre-trained model weights file (``.pth``).
    n_channels : int
        Number of EEG channels in the data.
    n_classes : int, optional
        Number of classification classes.  Defaults to ``2``.

    Returns
    -------
    predictions : np.ndarray
        Predicted class indices, shape ``(n_trials,)``.
    probabilities : np.ndarray
        Softmax probabilities for each class, shape ``(n_trials, n_classes)``.

    Notes
    -----
    If *model_path* does not exist or cannot be loaded, the function falls back
    to a randomly-initialised model and emits a warning.  This allows the UI to
    remain functional for demonstration purposes without a trained model.
    """
    model = SimpleEEGNet(n_channels=n_channels, n_classes=n_classes)

    try:
        model.load_state_dict(
            torch.load(model_path, map_location=torch.device('cpu'))
        )
    except Exception as exc:
        print(
            f"Warning: Could not load weights from '{model_path}'. "
            f"Using untrained model. Error: {exc}"
        )

    model.eval()

    # EEGNet expects 4-D input: (batch_size, 1, channels, time_steps)
    tensor_data = torch.FloatTensor(data_array).unsqueeze(1)

    outputs = model(tensor_data)

    probabilities = torch.softmax(outputs, dim=1).numpy()
    predictions = np.argmax(probabilities, axis=1)

    return predictions, probabilities