"""EEG analysis utilities.

Provides functions for frequency analysis (PSD), ERP computation,
time-frequency analysis, and band power extraction.
"""

from typing import Dict, Final, List, Optional, Tuple

import mne
import numpy as np
from scipy import signal


# Standard EEG frequency bands (Hz)
FREQ_BANDS: Final[Dict[str, Tuple[float, float]]] = {
    "Delta": (0.5, 4.0),
    "Theta": (4.0, 8.0),
    "Alpha": (8.0, 13.0),
    "Beta": (13.0, 30.0),
    "Gamma": (30.0, 100.0),
}


def get_freq_bands() -> Dict[str, Tuple[float, float]]:
    """Return the standard EEG frequency band definitions."""
    return FREQ_BANDS.copy()


def compute_psd(
    raw: mne.io.BaseRaw,
    fmin: float = 0.5,
    fmax: float = 50.0,
    method: str = "welch",
) -> Tuple[np.ndarray, np.ndarray]:
    """Compute Power Spectral Density (PSD) of the EEG data.

    Parameters
    ----------
    raw : mne.io.Raw
        The raw EEG data.
    fmin : float
        Minimum frequency of interest in Hz.
    fmax : float
        Maximum frequency of interest in Hz.
    method : str
        PSD estimation method: 'welch' or 'multitaper'.

    Returns
    -------
    psd_data : np.ndarray
        PSD values, shape (n_channels, n_freqs).
    freqs : np.ndarray
        Frequency values in Hz.
    """
    spectrum = raw.compute_psd(method=method, fmin=fmin, fmax=fmax, verbose=False)
    psd_data = spectrum.get_data()
    freqs = spectrum.freqs
    return psd_data, freqs


def compute_band_power(
    raw: mne.io.BaseRaw,
    bands: Optional[Dict[str, Tuple[float, float]]] = None,
) -> Dict[str, np.ndarray]:
    """Compute average power in standard EEG frequency bands.

    Parameters
    ----------
    raw : mne.io.Raw
        The raw EEG data.
    bands : dict or None
        Dictionary mapping band names to (fmin, fmax) tuples.
        If None, uses standard EEG bands.

    Returns
    -------
    band_powers : dict
        Dictionary mapping band names to arrays of power values
        per channel, shape (n_channels,).
    """
    if bands is None:
        bands = FREQ_BANDS

    band_powers = {}
    for band_name, (fmin, fmax) in bands.items():
        psd_data, freqs = compute_psd(raw, fmin=fmin, fmax=fmax)
        # Average power across frequencies for each channel
        band_powers[band_name] = np.mean(psd_data, axis=1)

    return band_powers


def compute_erp(
    raw: mne.io.BaseRaw,
    events: np.ndarray,
    event_id: Dict[str, int],
    tmin: float = -0.2,
    tmax: float = 0.8,
) -> Dict[str, mne.Evoked]:
    """Compute Event-Related Potentials (ERPs).

    Parameters
    ----------
    raw : mne.io.Raw
        The raw EEG data.
    events : np.ndarray
        Events array, shape (n_events, 3).
    event_id : dict
        Dictionary mapping event names to event codes.
    tmin : float
        Start time of the epoch in seconds (relative to event).
    tmax : float
        End time of the epoch in seconds (relative to event).

    Returns
    -------
    evoked_dict : dict
        Dictionary mapping event names to mne.Evoked objects.
    """
    epochs = mne.Epochs(
        raw, events, event_id,
        tmin=tmin, tmax=tmax,
        baseline=(tmin, 0),
        preload=True,
        verbose=False,
    )

    evoked_dict = {}
    for name in event_id:
        evoked_dict[name] = epochs[name].average()

    return evoked_dict


def compute_tfr(
    raw: mne.io.BaseRaw,
    freqs: Optional[np.ndarray] = None,
    n_cycles: Optional[np.ndarray] = None,
    method: str = "morlet",
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compute Time-Frequency Representation using Morlet wavelets.

    Parameters
    ----------
    raw : mne.io.Raw
        The raw EEG data.
    freqs : np.ndarray or None
        Frequencies of interest. If None, uses 1-40 Hz.
    n_cycles : np.ndarray or None
        Number of cycles per frequency. If None, uses freqs / 2.
    method : str
        TFR method. Currently supports 'morlet'.

    Returns
    -------
    power : np.ndarray
        Time-frequency power, shape (n_channels, n_freqs, n_times).
    times : np.ndarray
        Time points in seconds.
    freqs_out : np.ndarray
        Frequency values in Hz.
    """
    if freqs is None:
        freqs = np.arange(1, 41, 1.0)
    if n_cycles is None:
        n_cycles = freqs / 2.0

    data = raw.get_data()
    sfreq = raw.info["sfreq"]
    n_channels, n_times = data.shape

    power = np.zeros((n_channels, len(freqs), n_times))

    for i, (freq, nc) in enumerate(zip(freqs, n_cycles)):
        # Create Morlet wavelet
        sigma_t = nc / (2.0 * np.pi * freq)
        t_wavelet = np.arange(-4 * sigma_t, 4 * sigma_t, 1.0 / sfreq)
        wavelet = np.exp(2j * np.pi * freq * t_wavelet) * np.exp(
            -(t_wavelet**2) / (2 * sigma_t**2)
        )

        for ch in range(n_channels):
            analytic = signal.fftconvolve(data[ch], wavelet, mode="same")
            power[ch, i, :] = np.abs(analytic) ** 2

    times = np.arange(n_times) / sfreq
    return power, times, freqs


def compute_connectivity(
    raw: mne.io.BaseRaw,
    method: str = "correlation",
) -> Tuple[np.ndarray, List[str]]:
    """Compute channel connectivity matrix.

    Parameters
    ----------
    raw : mne.io.Raw
        The raw EEG data.
    method : str
        Connectivity method. Supports 'correlation'.

    Returns
    -------
    conn_matrix : np.ndarray
        Connectivity matrix, shape (n_channels, n_channels).
    ch_names : list
        Channel names corresponding to matrix indices.
    """
    data = raw.get_data()

    if method == "correlation":
        conn_matrix = np.corrcoef(data)
    else:
        raise ValueError(
            f"Unsupported method: '{method}'. Supported: ['correlation']"
        )

    return conn_matrix, raw.ch_names