"""EEG data loading utilities.

Supports loading EEG data from various file formats including
EDF, BDF, FIF, SET, CSV, and BrainVision files using the MNE library.
"""

import os
import tempfile

import mne
import numpy as np
import pandas as pd


SUPPORTED_FORMATS = {
    ".edf": "EDF (European Data Format)",
    ".bdf": "BDF (BioSemi Data Format)",
    ".fif": "FIF (MNE-Python native)",
    ".set": "SET (EEGLAB)",
    ".csv": "CSV (Comma-Separated Values)",
    ".vhdr": "BrainVision (.vhdr + .eeg + .vmrk)",
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
    elif file_type == ".vhdr":
        raw = mne.io.read_raw_brainvision(file_path, preload=True)
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


def load_brainvision_files(vhdr_buffer, eeg_buffer, vmrk_buffer, basename="data"):
    """Load BrainVision data from in-memory file buffers.

    BrainVision format consists of three files (.vhdr, .eeg, .vmrk) that must
    reside in the same directory. This function writes the buffers to a
    temporary directory and loads them with MNE.

    Parameters
    ----------
    vhdr_buffer : bytes or buffer
        Content of the .vhdr header file.
    eeg_buffer : bytes or buffer
        Content of the .eeg data file.
    vmrk_buffer : bytes or buffer
        Content of the .vmrk marker file.
    basename : str
        Base filename used for the temporary files (without extension).

    Returns
    -------
    raw : mne.io.Raw
        The loaded raw EEG data.
    """
    tmp_dir = tempfile.mkdtemp()
    try:
        vhdr_path = os.path.join(tmp_dir, f"{basename}.vhdr")
        eeg_path = os.path.join(tmp_dir, f"{basename}.eeg")
        vmrk_path = os.path.join(tmp_dir, f"{basename}.vmrk")

        # Read header content and rewrite DataFile / MarkerFile references
        # so they point to the co-located temp files.
        vhdr_content = (
            bytes(vhdr_buffer) if not isinstance(vhdr_buffer, bytes)
            else vhdr_buffer
        )
        vhdr_text = vhdr_content.decode("utf-8", errors="replace")
        vhdr_text = _rewrite_brainvision_header(
            vhdr_text, f"{basename}.eeg", f"{basename}.vmrk"
        )

        with open(vhdr_path, "w", encoding="utf-8") as f:
            f.write(vhdr_text)
        with open(eeg_path, "wb") as f:
            f.write(bytes(eeg_buffer) if not isinstance(eeg_buffer, bytes)
                    else eeg_buffer)
        with open(vmrk_path, "wb") as f:
            f.write(bytes(vmrk_buffer) if not isinstance(vmrk_buffer, bytes)
                    else vmrk_buffer)

        raw = mne.io.read_raw_brainvision(vhdr_path, preload=True)
    finally:
        # Clean up temp files
        for path in (vhdr_path, eeg_path, vmrk_path):
            if os.path.exists(path):
                os.unlink(path)
        os.rmdir(tmp_dir)

    return raw


def _rewrite_brainvision_header(vhdr_text, eeg_filename, vmrk_filename):
    """Rewrite DataFile and MarkerFile references in a .vhdr header string.

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
        Updated .vhdr content.
    """
    import re
    vhdr_text = re.sub(
        r"(?m)^DataFile=.*$", f"DataFile={eeg_filename}", vhdr_text
    )
    vhdr_text = re.sub(
        r"(?m)^MarkerFile=.*$", f"MarkerFile={vmrk_filename}", vhdr_text
    )
    return vhdr_text
