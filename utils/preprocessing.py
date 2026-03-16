"""EEG preprocessing utilities.

Provides functions for filtering, re-referencing, artifact removal,
and other common EEG preprocessing steps using MNE.
"""

import mne
import numpy as np


def apply_bandpass_filter(raw, l_freq=0.1, h_freq=40.0):
    """Apply a bandpass filter to the raw EEG data.

    Parameters
    ----------
    raw : mne.io.Raw
        The raw EEG data (will be copied, not modified in place).
    l_freq : float
        Low cutoff frequency in Hz.
    h_freq : float
        High cutoff frequency in Hz.

    Returns
    -------
    raw_filtered : mne.io.Raw
        The filtered raw data.
    """
    raw_filtered = raw.copy().filter(l_freq=l_freq, h_freq=h_freq, verbose=False)
    return raw_filtered


def apply_notch_filter(raw, freqs=50.0):
    """Apply a notch filter to remove power line noise.

    Parameters
    ----------
    raw : mne.io.Raw
        The raw EEG data (will be copied, not modified in place).
    freqs : float or array-like
        Frequency/frequencies to notch filter in Hz.
        Common values: 50.0 (Europe/Asia) or 60.0 (Americas).

    Returns
    -------
    raw_notched : mne.io.Raw
        The notch-filtered raw data.
    """
    if isinstance(freqs, (int, float)):
        freqs = [freqs]
    raw_notched = raw.copy().notch_filter(freqs=freqs, verbose=False)
    return raw_notched


def apply_rereferencing(raw, ref_type="average"):
    """Apply re-referencing to the EEG data.

    Parameters
    ----------
    raw : mne.io.Raw
        The raw EEG data (will be copied, not modified in place).
    ref_type : str
        Type of reference. Options: 'average' or a channel name.

    Returns
    -------
    raw_reref : mne.io.Raw
        The re-referenced raw data.
    """
    raw_reref = raw.copy()
    if ref_type == "average":
        raw_reref.set_eeg_reference("average", projection=False, verbose=False)
    else:
        raw_reref.set_eeg_reference([ref_type], verbose=False)
    return raw_reref


def run_ica(raw, n_components=None, random_state=42):
    """Run Independent Component Analysis (ICA) on the data.

    Parameters
    ----------
    raw : mne.io.Raw
        The raw EEG data.
    n_components : int or None
        Number of ICA components. If None, uses number of channels.
    random_state : int
        Random seed for reproducibility.

    Returns
    -------
    ica : mne.preprocessing.ICA
        The fitted ICA object.
    """
    if n_components is None:
        n_components = min(len(raw.ch_names), 20)
    ica = mne.preprocessing.ICA(
        n_components=n_components,
        random_state=random_state,
        max_iter="auto",
    )
    ica.fit(raw, verbose=False)
    return ica


def apply_ica_exclusion(raw, ica, exclude_idx):
    """Apply ICA with specified components excluded.

    Parameters
    ----------
    raw : mne.io.Raw
        The raw EEG data.
    ica : mne.preprocessing.ICA
        A fitted ICA object.
    exclude_idx : list of int
        Indices of ICA components to exclude.

    Returns
    -------
    raw_clean : mne.io.Raw
        The cleaned raw data with specified ICA components removed.
    """
    ica.exclude = list(exclude_idx)
    raw_clean = raw.copy()
    ica.apply(raw_clean, verbose=False)
    return raw_clean


def detect_bad_channels(raw, method="zscore", threshold=3.0):
    """Detect bad channels using statistical methods.

    Parameters
    ----------
    raw : mne.io.Raw
        The raw EEG data.
    method : str
        Detection method. Currently supports 'zscore'.
    threshold : float
        Z-score threshold for marking a channel as bad.

    Returns
    -------
    bad_channels : list of str
        List of channel names detected as bad.
    """
    data = raw.get_data()

    if method == "zscore":
        ch_std = np.std(data, axis=1)
        mean_std = np.mean(ch_std)
        std_std = np.std(ch_std)

        if std_std == 0:
            return []

        z_scores = np.abs((ch_std - mean_std) / std_std)
        bad_idx = np.where(z_scores > threshold)[0]
        bad_channels = [raw.ch_names[i] for i in bad_idx]
    else:
        raise ValueError(f"Unsupported method: '{method}'. Supported: ['zscore']")

    return bad_channels