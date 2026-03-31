"""
MNE Utilities - Helper functions for MNE compatibility.

Handles type checking for all MNE Raw formats.
"""

import mne


def is_mne_raw(obj) -> bool:
    """
    Check if object is any type of MNE Raw object.

    Parameters
    ----------
    obj : Any
        Object to check.

    Returns
    -------
    bool
        True if object is an MNE Raw object (any subclass).

    Notes
    -----
    This function handles all MNE Raw types:
    - mne.io.Raw
    - mne.io.RawEDF
    - mne.io.RawBDF
    - mne.io.RawFIF
    - mne.io.RawEEGLab
    - mne.io.RawBrainVision
    - mne.io.RawArray
    - Any other mne.io.BaseRaw subclass
    """
    try:
        # Check if it's a BaseRaw instance (parent class of all MNE Raw types)
        return isinstance(obj, mne.io.BaseRaw)
    except (AttributeError, TypeError):
        return False


def is_mne_ica(obj) -> bool:
    """
    Check if object is an MNE ICA object.

    Parameters
    ----------
    obj : Any
        Object to check.

    Returns
    -------
    bool
        True if object is an MNE ICA object.
    """
    try:
        return isinstance(obj, mne.preprocessing.ICA)
    except (AttributeError, TypeError):
        return False


def get_raw_type_name(raw) -> str:
    """
    Get a friendly name for the raw data type.

    Parameters
    ----------
    raw : mne.io.BaseRaw
        Raw data object.

    Returns
    -------
    str
        Friendly type name (e.g., "BrainVision", "EDF", "FIF").
    """
    type_name = type(raw).__name__

    # Map internal names to friendly names
    type_mapping = {
        "RawEDF": "EDF",
        "RawBDF": "BioSemi",
        "RawFIF": "FIF",
        "RawEEGLab": "EEGLAB",
        "RawBrainVision": "BrainVision",
        "RawArray": "Array",
        "Raw": "Generic",
    }

    return type_mapping.get(type_name, type_name)
