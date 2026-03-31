"""EEG preprocessing utilities.

Provides functions for filtering, re-referencing, artifact removal,
and other common EEG preprocessing steps using MNE.

Supports all MNE Raw formats: EDF, BDF, FIF, EEGLAB, BrainVision, CSV, Array.
"""

from typing import List, Optional, Union
import mne
import numpy as np

from config import (
    BANDPASS_L_FREQ,
    BANDPASS_H_FREQ,
    NOTCH_FREQS_EU_ASIA,
    NOTCH_FREQS_AMERICAS,
    ICA_DEFAULT_N_COMPONENTS,
    ICA_MAX_ITERATIONS,
    ICA_RANDOM_STATE,
    BAD_CHANNEL_ZSCORE_THRESHOLD,
    BAD_CHANNEL_DETECTION_METHOD,
)
from utils.logger import get_logger
from utils.mne_utils import is_mne_raw, is_mne_ica

logger = get_logger("preprocessing")


def apply_bandpass_filter(
    raw, l_freq: float = BANDPASS_L_FREQ, h_freq: float = BANDPASS_H_FREQ
):
    """Apply a bandpass filter to the raw EEG data.

    Supports all MNE Raw formats.

    Parameters
    ----------
    raw : mne.io.BaseRaw
        The raw EEG data (any format: EDF, BDF, FIF, EEGLAB, BrainVision, etc).
        Will be copied, not modified in place.
    l_freq : float
        Low cutoff frequency in Hz. Default is 0.1.
    h_freq : float
        High cutoff frequency in Hz. Default is 40.0.

    Returns
    -------
    mne.io.BaseRaw
        The filtered raw data (same type as input).

    Raises
    ------
    TypeError
        If raw is not an MNE Raw object.
    ValueError
        If l_freq >= h_freq or frequencies are invalid.

    Examples
    --------
    >>> raw_filtered = apply_bandpass_filter(raw, l_freq=0.1, h_freq=40.0)
    """
    # Type validation - accepts all MNE Raw types
    if not is_mne_raw(raw):
        raw_type = type(raw).__name__
        raise TypeError(
            f"raw must be MNE Raw object, got {raw_type}. "
            f"Supported formats: EDF, BDF, FIF, EEGLAB, BrainVision, Array, CSV"
        )

    # Parameter validation
    if l_freq >= h_freq:
        raise ValueError(f"l_freq ({l_freq}) must be < h_freq ({h_freq})")

    if l_freq < 0:
        raise ValueError(f"l_freq must be >= 0, got {l_freq}")

    # Check Nyquist frequency
    nyquist = raw.info["sfreq"] / 2
    if h_freq > nyquist:
        raise ValueError(f"h_freq ({h_freq}) exceeds Nyquist frequency ({nyquist})")

    try:
        logger.info(f"Applying bandpass filter: {l_freq}-{h_freq} Hz")
        raw_filtered = raw.copy().filter(l_freq=l_freq, h_freq=h_freq, verbose=False)
        logger.debug("Bandpass filter applied successfully")
        return raw_filtered

    except Exception as e:
        logger.error(f"Bandpass filter failed: {str(e)}")
        raise ValueError(f"Filter application failed: {str(e)}") from e


def apply_notch_filter(raw, freqs: Union[float, List[float]] = NOTCH_FREQS_EU_ASIA):
    """Apply a notch filter to remove power line noise.

    Supports all MNE Raw formats.

    Parameters
    ----------
    raw : mne.io.BaseRaw
        The raw EEG data (any format).
        Will be copied, not modified in place.
    freqs : float or list of float
        Frequency/frequencies to notch filter in Hz.
        Common values: 50.0 (Europe/Asia) or 60.0 (Americas).
        Default is 50.0.

    Returns
    -------
    mne.io.BaseRaw
        The notch-filtered raw data (same type as input).

    Raises
    ------
    TypeError
        If raw is not an MNE Raw object or freqs is wrong type.
    ValueError
        If frequencies are invalid.

    Examples
    --------
    >>> raw_notched = apply_notch_filter(raw, freqs=50.0)
    >>> raw_notched = apply_notch_filter(raw, freqs=[50.0, 100.0])
    """
    # Type validation
    if not is_mne_raw(raw):
        raw_type = type(raw).__name__
        raise TypeError(
            f"raw must be MNE Raw object, got {raw_type}. "
            f"Supported formats: EDF, BDF, FIF, EEGLAB, BrainVision, Array, CSV"
        )

    # Convert single frequency to list
    if isinstance(freqs, (int, float)):
        freqs = [freqs]
    elif not isinstance(freqs, list):
        raise TypeError(f"freqs must be float or list, got {type(freqs).__name__}")

    # Validate frequencies
    for freq in freqs:
        if freq <= 0:
            raise ValueError(f"Frequencies must be > 0, got {freq}")
        nyquist = raw.info["sfreq"] / 2
        if freq > nyquist:
            raise ValueError(f"Frequency {freq} exceeds Nyquist {nyquist}")

    try:
        logger.info(f"Applying notch filter at: {freqs} Hz")
        raw_notched = raw.copy().notch_filter(freqs=freqs, verbose=False)
        logger.debug("Notch filter applied successfully")
        return raw_notched

    except Exception as e:
        logger.error(f"Notch filter failed: {str(e)}")
        raise ValueError(f"Filter application failed: {str(e)}") from e


def apply_rereferencing(raw, ref_type: str = "average"):
    """Apply re-referencing to the EEG data.

    Supports all MNE Raw formats.

    Parameters
    ----------
    raw : mne.io.BaseRaw
        The raw EEG data (any format).
        Will be copied, not modified in place.
    ref_type : str
        Type of reference. Options: 'average' or a channel name.
        Default is 'average'.

    Returns
    -------
    mne.io.BaseRaw
        The re-referenced raw data (same type as input).

    Raises
    ------
    TypeError
        If raw is not an MNE Raw object.
    ValueError
        If reference channel does not exist.

    Examples
    --------
    >>> raw_reref = apply_rereferencing(raw, ref_type="average")
    >>> raw_reref = apply_rereferencing(raw, ref_type="Cz")
    """
    # Type validation
    if not is_mne_raw(raw):
        raw_type = type(raw).__name__
        raise TypeError(
            f"raw must be MNE Raw object, got {raw_type}. "
            f"Supported formats: EDF, BDF, FIF, EEGLAB, BrainVision, Array, CSV"
        )

    # Validate reference type
    if ref_type not in ["average"] and ref_type not in raw.ch_names:
        raise ValueError(
            f"Reference channel '{ref_type}' not found. "
            f"Available channels: {raw.ch_names}"
        )

    try:
        logger.info(f"Applying re-referencing: {ref_type}")
        raw_reref = raw.copy()

        if ref_type == "average":
            raw_reref.set_eeg_reference("average", projection=False, verbose=False)
        else:
            raw_reref.set_eeg_reference([ref_type], verbose=False)

        logger.debug(f"Re-referencing applied: {ref_type}")
        return raw_reref

    except Exception as e:
        logger.error(f"Re-referencing failed: {str(e)}")
        raise ValueError(f"Re-referencing failed: {str(e)}") from e


def run_ica(
    raw, n_components: Optional[int] = None, random_state: int = ICA_RANDOM_STATE
) -> mne.preprocessing.ICA:
    """Run Independent Component Analysis (ICA) on the data.

    Supports all MNE Raw formats.
    Results are cached to avoid recomputation on identical inputs.

    Parameters
    ----------
    raw : mne.io.BaseRaw
        The raw EEG data (any format).
    n_components : int or None
        Number of ICA components. If None, uses min(n_channels, 20).
    random_state : int
        Random seed for reproducibility. Default is 42.

    Returns
    -------
    mne.preprocessing.ICA
        The fitted ICA object.

    Raises
    ------
    TypeError
        If raw is not an MNE Raw object.
    ValueError
        If n_components is invalid.

    Examples
    --------
    >>> ica = run_ica(raw, n_components=20)
    """
    # Type validation
    if not is_mne_raw(raw):
        raw_type = type(raw).__name__
        raise TypeError(
            f"raw must be MNE Raw object, got {raw_type}. "
            f"Supported formats: EDF, BDF, FIF, EEGLAB, BrainVision, Array, CSV"
        )

    # Set default n_components
    if n_components is None:
        n_components = min(len(raw.ch_names), ICA_DEFAULT_N_COMPONENTS)
        logger.debug(f"Set n_components to {n_components}")

    # Validate n_components
    if not isinstance(n_components, int) or n_components <= 0:
        raise ValueError(f"n_components must be positive int, got {n_components}")

    if n_components > len(raw.ch_names):
        raise ValueError(
            f"n_components ({n_components}) cannot exceed "
            f"number of channels ({len(raw.ch_names)})"
        )

    try:
        logger.info(f"Running ICA with {n_components} components")
        ica = mne.preprocessing.ICA(
            n_components=n_components,
            random_state=random_state,
            max_iter=ICA_MAX_ITERATIONS,
            verbose=False,
        )
        ica.fit(raw, verbose=False)
        logger.info("ICA fitting completed successfully")
        return ica

    except Exception as e:
        logger.error(f"ICA fitting failed: {str(e)}")
        raise ValueError(f"ICA fitting failed: {str(e)}") from e


def apply_ica_exclusion(raw, ica: mne.preprocessing.ICA, exclude_idx: List[int]):
    """Apply ICA with specified components excluded.

    Supports all MNE Raw formats.

    Parameters
    ----------
    raw : mne.io.BaseRaw
        The raw EEG data (any format).
    ica : mne.preprocessing.ICA
        A fitted ICA object.
    exclude_idx : list of int
        Indices of ICA components to exclude.

    Returns
    -------
    mne.io.BaseRaw
        The cleaned raw data with specified ICA components removed (same type as input).

    Raises
    ------
    TypeError
        If parameters are not the correct types.
    ValueError
        If component indices are invalid.

    Examples
    --------
    >>> raw_clean = apply_ica_exclusion(raw, ica, exclude_idx=[0, 2, 5])
    """
    # Type validation
    if not is_mne_raw(raw):
        raw_type = type(raw).__name__
        raise TypeError(
            f"raw must be MNE Raw object, got {raw_type}. "
            f"Supported formats: EDF, BDF, FIF, EEGLAB, BrainVision, Array, CSV"
        )

    if not is_mne_ica(ica):
        raise TypeError(f"ica must be mne.preprocessing.ICA, got {type(ica).__name__}")

    if not isinstance(exclude_idx, list):
        raise TypeError(f"exclude_idx must be list, got {type(exclude_idx).__name__}")

    # Validate indices
    for idx in exclude_idx:
        if not isinstance(idx, int):
            raise ValueError(f"Component indices must be integers, got {idx}")
        if idx < 0 or idx >= ica.n_components:
            raise ValueError(
                f"Component index {idx} out of range [0, {ica.n_components-1}]"
            )

    try:
        logger.info(f"Excluding ICA components: {exclude_idx}")
        ica.exclude = list(exclude_idx)
        raw_clean = raw.copy()
        ica.apply(raw_clean, verbose=False)
        logger.debug(f"Applied ICA exclusion: {len(exclude_idx)} components removed")
        return raw_clean

    except Exception as e:
        logger.error(f"ICA exclusion failed: {str(e)}")
        raise ValueError(f"ICA exclusion failed: {str(e)}") from e


def detect_bad_channels(
    raw,
    method: str = BAD_CHANNEL_DETECTION_METHOD,
    threshold: float = BAD_CHANNEL_ZSCORE_THRESHOLD,
) -> List[str]:
    """Detect bad channels using statistical methods.

    Supports all MNE Raw formats.

    Parameters
    ----------
    raw : mne.io.BaseRaw
        The raw EEG data (any format).
    method : str
        Detection method. Currently supports 'zscore'. Default is 'zscore'.
    threshold : float
        Z-score threshold for marking a channel as bad. Default is 3.0.

    Returns
    -------
    list of str
        List of channel names detected as bad.

    Raises
    ------
    TypeError
        If raw is not an MNE Raw object.
    ValueError
        If method is unsupported or threshold is invalid.

    Examples
    --------
    >>> bad_chs = detect_bad_channels(raw, method="zscore", threshold=3.0)
    """
    # Type validation
    if not is_mne_raw(raw):
        raw_type = type(raw).__name__
        raise TypeError(
            f"raw must be MNE Raw object, got {raw_type}. "
            f"Supported formats: EDF, BDF, FIF, EEGLAB, BrainVision, Array, CSV"
        )

    if method not in ["zscore"]:
        raise ValueError(f"Unsupported method: '{method}'. Supported: ['zscore']")

    if threshold <= 0:
        raise ValueError(f"threshold must be > 0, got {threshold}")

    try:
        logger.info(f"Detecting bad channels using {method} (threshold={threshold})")
        data = raw.get_data()

        if method == "zscore":
            # Calculate standard deviation per channel
            ch_std = np.std(data, axis=1)
            mean_std = np.mean(ch_std)
            std_std = np.std(ch_std)

            # Handle zero variance
            if std_std == 0:
                logger.debug("All channels have identical variance")
                return []

            # Calculate z-scores
            z_scores = np.abs((ch_std - mean_std) / std_std)
            bad_idx = np.where(z_scores > threshold)[0]
            bad_channels = [raw.ch_names[i] for i in bad_idx]

            if bad_channels:
                logger.warning(
                    f"Detected {len(bad_channels)} bad channels: {bad_channels}"
                )
            else:
                logger.debug("No bad channels detected")

            return bad_channels

    except Exception as e:
        logger.error(f"Bad channel detection failed: {str(e)}")
        raise ValueError(f"Bad channel detection failed: {str(e)}") from e
