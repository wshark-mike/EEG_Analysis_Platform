"""
Global configuration for EEG Analysis Platform.

Centralized settings for all EEG processing parameters, making the codebase
more maintainable and easier to modify without changing function signatures.
"""

# ===== Data Loading Configuration =====
DEFAULT_CSV_SFREQ = 250.0  # Default sampling frequency for CSV files (Hz)
CSV_MICROVOLTS_THRESHOLD = 1.0  # Threshold for auto-scaling microvolts to volts

# ===== Bandpass Filter Configuration =====
BANDPASS_L_FREQ = 0.1  # Low cutoff frequency (Hz)
BANDPASS_H_FREQ = 40.0  # High cutoff frequency (Hz)
BANDPASS_FILTER_ORDER = 4  # Filter order for butterworth filter

# ===== Notch Filter Configuration =====
NOTCH_FREQS_EU_ASIA = 50.0  # Power line frequency in Europe/Asia (Hz)
NOTCH_FREQS_AMERICAS = 60.0  # Power line frequency in Americas (Hz)
NOTCH_FILTER_WIDTH = 2.0  # Bandwidth of notch filter (Hz)

# ===== ICA Configuration =====
ICA_DEFAULT_N_COMPONENTS = 20  # Default number of ICA components
ICA_MAX_ITERATIONS = "auto"  # Maximum iterations for ICA fitting
ICA_RANDOM_STATE = 42  # Random seed for reproducibility

# ===== Bad Channel Detection =====
BAD_CHANNEL_ZSCORE_THRESHOLD = 3.0  # Z-score threshold for bad channel detection
BAD_CHANNEL_DETECTION_METHOD = "zscore"  # Detection method

# ===== Visualization Configuration =====
PLOT_MAX_POINTS_PER_TRACE = 2000  # Maximum points to display per trace (Plotly)
PLOT_SIGNAL_LINEWIDTH = 0.5  # Line width for signal plots
PLOT_FIGURE_HEIGHT_PER_CHANNEL = 200  # Height per channel in Plotly plots (pixels)
PLOT_MIN_FIGURE_HEIGHT = 400  # Minimum figure height (pixels)
PLOT_MATPLOTLIB_DPI = 100  # DPI for matplotlib figures

# ===== PSD Analysis Configuration =====
PSD_METHOD = "welch"  # Method for PSD computation ('welch' or 'multitaper')
PSD_FMIN = 0.5  # Minimum frequency for PSD (Hz)
PSD_FMAX = 50.0  # Maximum frequency for PSD (Hz)

# ===== Frequency Bands =====
FREQUENCY_BANDS = {
    "Delta": (0.5, 4),
    "Theta": (4, 8),
    "Alpha": (8, 12),
    "Beta": (12, 30),
    "Gamma": (30, 50),
}

# ===== Logging Configuration =====
LOG_LEVEL = "INFO"  # Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
LOG_FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
LOG_FILE = "eeg_analysis.log"  # Log file name

# ===== Cache Configuration =====
CACHE_MAX_SIZE = 3  # Maximum number of cached results per function
CACHE_TTL_SECONDS = 3600  # Cache time-to-live (1 hour)

# ===== Performance Thresholds =====
MAX_MEMORY_MB = 2048  # Maximum memory usage (MB)
PROCESSING_TIMEOUT_SECONDS = 600  # Processing timeout (10 minutes)

# ===== Session & Memory Management =====
MAX_HISTORY_DEPTH = 5  # Maximum undo history depth to prevent memory leaks
MAX_SESSION_MEMORY_MB = 2048  # Maximum session memory in MB

# ===== File Upload Configuration =====
MAX_FILE_SIZE_MB = 500  # Maximum file upload size in MB
ALLOWED_UPLOAD_FORMATS = ["edf", "bdf", "fif", "set", "csv", "vhdr", "eeg", "vmrk"]

# ===== API & Security Configuration =====
import os
from dotenv import load_dotenv

load_dotenv()

# LLM Provider Selection
LLM_PROVIDER = os.getenv("LLM_PROVIDER", "gemini").lower()  # "openai" or "gemini" (default: gemini)

# OpenAI Configuration
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")

# Google Gemini Configuration
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")

# API Usage Limits
MAX_API_TOKENS_PER_SESSION = 10000  # ~$0.10 max cost per session

# ===== EEGNet Model Configuration =====
EEGNET_TIME_STRIDE = 32  # Time dimension stride (from model pooling design)
EEGNET_POOL_SIZE = 4  # Average pooling size
EEGNET_TYPICAL_SFREQ = 256  # Typical sampling rate (Hz)
