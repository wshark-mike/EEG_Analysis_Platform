"""EEG data loading utilities.

Supports loading EEG data from various file formats including
EDF, BDF, FIF, SET, and CSV files using the MNE library.
"""

import mne
import numpy as np
import pandas as pd


SUPPORTED_FORMATS = {
    ".edf": "EDF (European Data Format)",
    ".bdf": "BDF (BioSemi Data Format)",
    ".fif": "FIF (MNE-Python native)",
    ".set": "SET (EEGLAB)",
    ".csv": "CSV (Comma-Separated Values)",
}


def get_supported_formats():
    """Return a dictionary of supported file formats and their descriptions."""
    return SUPPORTED_FORMATS.copy()


def load_eeg_file(file_path, file_type=None):
    """Load an EEG data file and return an MNE Raw object.

    Parameters
    ----------
    file_path : str
        Path to the EEG data file.
    file_type : str, optional
        File extension (e.g., '.edf'). If None, inferred from file_path.

    Returns
    -------
    raw : mne.io.Raw
        The loaded raw EEG data.

    Raises
    ------
    ValueError
        If the file format is not supported.
    """
    if file_type is None:
        import os
        file_type = os.path.splitext(file_path)[1].lower()

    if file_type == ".edf":
        raw = mne.io.read_raw_edf(file_path, preload=True)
    elif file_type == ".bdf":
        raw = mne.io.read_raw_bdf(file_path, preload=True)
    elif file_type == ".fif":
        raw = mne.io.read_raw_fif(file_path, preload=True)
    elif file_type == ".set":
        raw = mne.io.read_raw_eeglab(file_path, preload=True)
    elif file_type == ".csv":
        raw = _load_csv_as_raw(file_path)
    else:
        raise ValueError(
            f"Unsupported file format: '{file_type}'. "
            f"Supported formats: {list(SUPPORTED_FORMATS.keys())}"
        )
    return raw


def _load_csv_as_raw(file_path, sfreq=256.0):
    """Load a CSV file as an MNE Raw object.

    Expects CSV with columns as channel names and rows as time samples.

    Parameters
    ----------
    file_path : str
        Path to the CSV file.
    sfreq : float
        Sampling frequency in Hz. Default is 256.0.

    Returns
    -------
    raw : mne.io.RawArray
        The loaded raw EEG data.
    """
    df = pd.read_csv(file_path)

    # If a 'time' or 'timestamp' column exists, drop it
    time_cols = [c for c in df.columns if c.lower() in ("time", "timestamp")]
    if time_cols:
        df = df.drop(columns=time_cols)

    ch_names = list(df.columns)
    data = df.values.T  # MNE expects (n_channels, n_times)

    # Scale to volts if data appears to be in microvolts
    if np.abs(data).max() > 1.0:
        data = data * 1e-6

    ch_types = ["eeg"] * len(ch_names)
    info = mne.create_info(ch_names=ch_names, sfreq=sfreq, ch_types=ch_types)
    raw = mne.io.RawArray(data, info)
    return raw


def get_raw_info(raw):
    """Extract basic information from a Raw object.

    Parameters
    ----------
    raw : mne.io.Raw
        The raw EEG data.

    Returns
    -------
    info_dict : dict
        Dictionary with EEG data information.
    """
    return {
        "n_channels": len(raw.ch_names),
        "ch_names": raw.ch_names,
        "sfreq": raw.info["sfreq"],
        "duration_sec": raw.n_times / raw.info["sfreq"],
        "n_samples": raw.n_times,
        "ch_types": [mne.channel_type(raw.info, i) for i in range(len(raw.ch_names))],
    }