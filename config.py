"""Centralized configuration constants for the EEG Analysis Platform.

All default values and magic numbers should be defined here to provide
a single source of truth and simplify parameter tuning.
"""

# ---------------------------------------------------------------------------
# Data loading defaults
# ---------------------------------------------------------------------------

#: Default sampling frequency (Hz) assumed when loading CSV files
CSV_DEFAULT_SFREQ: float = 256.0

# ---------------------------------------------------------------------------
# Filtering defaults
# ---------------------------------------------------------------------------

#: Default notch filter frequency (Hz) — European/Asian power line noise
NOTCH_FREQ_DEFAULT: float = 50.0

#: Default bandpass lower cutoff frequency (Hz)
BANDPASS_L_FREQ: float = 0.1

#: Default bandpass upper cutoff frequency (Hz)
BANDPASS_H_FREQ: float = 40.0

# ---------------------------------------------------------------------------
# ICA defaults
# ---------------------------------------------------------------------------

#: Default number of ICA components (None means use number of EEG channels)
ICA_DEFAULT_N_COMPONENTS: int = 20

#: Random seed for ICA reproducibility
ICA_RANDOM_STATE: int = 42

# ---------------------------------------------------------------------------
# Bad-channel detection defaults
# ---------------------------------------------------------------------------

#: Z-score threshold above which a channel is considered bad
BAD_CHANNEL_ZSCORE_THRESHOLD: float = 3.0

# ---------------------------------------------------------------------------
# PSD / spectral analysis defaults
# ---------------------------------------------------------------------------

#: Default minimum frequency for PSD computation (Hz)
PSD_FMIN: float = 0.5

#: Default maximum frequency for PSD computation (Hz)
PSD_FMAX: float = 50.0

# ---------------------------------------------------------------------------
# Visualization defaults
# ---------------------------------------------------------------------------

#: Default duration of the signal window shown in plots (seconds)
VIZ_DEFAULT_DURATION: float = 10.0

#: Maximum data points per trace in interactive Plotly plots
VIZ_MAX_POINTS_PER_TRACE: int = 2000

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

#: Default log level for the platform logger
LOG_LEVEL: str = "INFO"

#: Log file path (None means log to stderr only)
LOG_FILE: str = "eeg_platform.log"
