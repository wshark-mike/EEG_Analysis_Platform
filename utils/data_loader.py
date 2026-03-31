"""EEG data loading utilities.

Supports loading EEG data from various file formats including
EDF, BDF, FIF, SET, CSV, and BrainVision files using the MNE library.
"""

import os
import re
import mne
import numpy as np
import pandas as pd
import streamlit as st
from typing import Optional, Dict, List, Tuple

from config import DEFAULT_CSV_SFREQ, CSV_MICROVOLTS_THRESHOLD

from utils.logger import get_logger
from utils.mne_utils import is_mne_raw

logger = get_logger("data_loader")


SUPPORTED_FORMATS: Dict[str, str] = {
    ".edf": "EDF (European Data Format)",
    ".bdf": "BDF (BioSemi Data Format)",
    ".fif": "FIF (MNE-Python native)",
    ".set": "SET (EEGLAB)",
    ".csv": "CSV (Comma-Separated Values)",
    ".vhdr": "BrainVision (.vhdr + .eeg + .vmrk)",
}


def get_supported_formats() -> Dict[str, str]:
    """Return a dictionary of supported file formats and their descriptions.

    Returns
    -------
    Dict[str, str]
        Mapping of file extensions to format descriptions.
    """
    return SUPPORTED_FORMATS.copy()


@st.cache_resource(show_spinner=False)
def load_eeg_file(file_path: str, file_type: Optional[str] = None) -> mne.io.Raw:
    """Load an EEG data file and return an MNE Raw object.

    Parameters
    ----------
    file_path : str
        Path to the EEG data file.
    file_type : str, optional
        File extension (e.g., '.edf'). If None, inferred from file_path.

    Returns
    -------
    mne.io.Raw
        The loaded raw EEG data.

    Raises
    ------
    FileNotFoundError
        If the specified file does not exist.
    ValueError
        If the file format is not supported or file is corrupted.
    TypeError
        If file_path is not a string.

    Examples
    --------
    >>> raw = load_eeg_file("data/subject_01.edf")
    >>> raw = load_eeg_file("data/subject_01.csv", file_type=".csv")
    """
    # Type validation
    if not isinstance(file_path, str):
        raise TypeError(f"file_path must be str, got {type(file_path).__name__}")

    # File existence check
    if not os.path.exists(file_path):
        logger.error(f"File not found: {file_path}")
        raise FileNotFoundError(f"File not found: {file_path}")

    # Infer file type if not provided
    if file_type is None:
        file_type = os.path.splitext(file_path)[1].lower()
        logger.debug(f"Inferred file type: {file_type}")

    # Format validation
    if file_type not in SUPPORTED_FORMATS:
        logger.error(
            f"Unsupported file format: '{file_type}'. "
            f"Supported: {list(SUPPORTED_FORMATS.keys())}"
        )
        raise ValueError(
            f"Unsupported file format: '{file_type}'. "
            f"Supported formats: {list(SUPPORTED_FORMATS.keys())}"
        )

    # Load file based on format
    try:
        logger.info(f"Loading {file_type} file: {file_path}")

        if file_type == ".edf":
            raw = mne.io.read_raw_edf(file_path, preload=True, verbose=False)
        elif file_type == ".bdf":
            raw = mne.io.read_raw_bdf(file_path, preload=True, verbose=False)
        elif file_type == ".fif":
            raw = mne.io.read_raw_fif(file_path, preload=True, verbose=False)
        elif file_type == ".set":
            raw = mne.io.read_raw_eeglab(file_path, preload=True, verbose=False)
        elif file_type == ".csv":
            raw = _load_csv_as_raw(file_path)
        elif file_type == ".vhdr":
            raw = mne.io.read_raw_brainvision(file_path, preload=True, verbose=False)
        else:
            # Should never reach here due to earlier validation
            raise ValueError(f"Unsupported file format: {file_type}")

        # Ensure data is preloaded
        if not raw.preload:
            raw.load_data()

        logger.info(
            f"Successfully loaded EEG file: {len(raw.ch_names)} channels, "
            f"{raw.n_times / raw.info['sfreq']:.1f}s duration"
        )
        return raw

    except FileNotFoundError as e:
        logger.error(f"File not found: {file_path}")
        raise
    except Exception as e:
        logger.error(f"Failed to load {file_type} file: {str(e)}")
        raise ValueError(f"Failed to load EEG file: {str(e)}") from e


def _load_csv_as_raw(
    file_path: str, sfreq: float = DEFAULT_CSV_SFREQ
) -> mne.io.RawArray:
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
    mne.io.RawArray
        The loaded raw EEG data.

    Raises
    ------
    FileNotFoundError
        If CSV file does not exist.
    ValueError
        If CSV is malformed or contains invalid data.

    Examples
    --------
    >>> raw = _load_csv_as_raw("data/eeg_data.csv", sfreq=256.0)
    """
    try:
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"CSV file not found: {file_path}")

        logger.debug(f"Reading CSV file: {file_path}")
        df = pd.read_csv(file_path)

        if df.empty:
            raise ValueError("CSV file is empty")

        # Drop time/timestamp columns if present
        time_cols = [c for c in df.columns if c.lower() in ("time", "timestamp")]
        if time_cols:
            logger.debug(f"Dropping time columns: {time_cols}")
            df = df.drop(columns=time_cols)

        ch_names = list(df.columns)
        if not ch_names:
            raise ValueError("CSV has no channel columns after removing time column")

        # MNE expects (n_channels, n_times)
        data = df.values.T

        # Validate data
        if not np.isfinite(data).all():
            logger.warning("CSV contains NaN or infinite values - replacing with 0")
            data = np.nan_to_num(data, nan=0.0, posinf=0.0, neginf=0.0)

        # Auto-scale from microvolts to volts if needed
        if np.abs(data).max() > CSV_MICROVOLTS_THRESHOLD:
            logger.debug("Auto-scaling data from microvolts to volts")
            data = data * 1e-6

        # Create channel info
        ch_types = ["eeg"] * len(ch_names)
        info = mne.create_info(ch_names=ch_names, sfreq=sfreq, ch_types=ch_types)
        raw = mne.io.RawArray(data, info, verbose=False)

        logger.info(f"CSV loaded: {len(ch_names)} channels, {data.shape[1]} samples")
        return raw

    except Exception as e:
        logger.error(f"Failed to load CSV: {str(e)}")
        raise


def get_raw_info(raw: mne.io.Raw) -> Dict[str, any]:
    """Extract basic information from a Raw object.

    Parameters
    ----------
    raw : mne.io.Raw
        The raw EEG data.

    Returns
    -------
    Dict[str, any]
        Dictionary with EEG data information including:
        - n_channels: Number of channels
        - ch_names: Channel names
        - sfreq: Sampling frequency
        - duration_sec: Recording duration in seconds
        - n_samples: Total number of samples
        - ch_types: Channel types

    Examples
    --------
    >>> info = get_raw_info(raw)
    >>> print(f"Duration: {info['duration_sec']:.1f}s")
    """
    if not is_mne_raw(raw):
        raise TypeError(f"raw must be MNE Raw object, got {type(raw).__name__}")

    try:
        ch_types = []
        for i in range(len(raw.ch_names)):
            try:
                ch_type = mne.channel_type(raw.info, i)
                ch_types.append(ch_type)
            except Exception:
                ch_types.append("unknown")

        info_dict = {
            "n_channels": len(raw.ch_names),
            "ch_names": raw.ch_names,
            "sfreq": raw.info["sfreq"],
            "duration_sec": raw.n_times / raw.info["sfreq"],
            "n_samples": raw.n_times,
            "ch_types": ch_types,
        }

        logger.debug(
            f"Raw info: {info_dict['n_channels']} channels, "
            f"{info_dict['duration_sec']:.1f}s duration"
        )
        return info_dict

    except Exception as e:
        logger.error(f"Failed to extract raw info: {str(e)}")
        raise


def _rewrite_vhdr_references(
    vhdr_text: str, eeg_filename: str, vmrk_filename: str
) -> str:
    """Rewrite DataFile and MarkerFile references in a .vhdr header string.

    This function updates the BrainVision header file references to point to
    new data and marker file locations.

    Parameters
    ----------
    vhdr_text : str
        Original .vhdr file content as a string.
    eeg_filename : str
        New filename for the DataFile reference.
    vmrk_filename : str
        New filename for the MarkerFile reference.

    Returns
    -------
    str
        Updated .vhdr content with new file references.

    Examples
    --------
    >>> new_vhdr = _rewrite_vhdr_references(vhdr_text, "data.eeg", "data.vmrk")
    """
    try:
        # Update DataFile reference
        vhdr_text = re.sub(r"(?m)^DataFile=.*$", f"DataFile={eeg_filename}", vhdr_text)

        # Update MarkerFile reference
        vhdr_text = re.sub(
            r"(?m)^MarkerFile=.*$", f"MarkerFile={vmrk_filename}", vhdr_text
        )

        logger.debug(f"Updated VHDR references: {eeg_filename}, {vmrk_filename}")
        return vhdr_text

    except Exception as e:
        logger.error(f"Failed to rewrite VHDR references: {str(e)}")
        raise
