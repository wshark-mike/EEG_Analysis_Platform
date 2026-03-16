# 🧠 EEG Analysis Platform

A Streamlit-based web application for electroencephalography (EEG) data analysis.
Upload, visualize, preprocess, and analyze EEG data through an intuitive browser interface.

## Features

- **📂 Data Upload** — Load EEG data from EDF, BDF, FIF, SET, and CSV files
- **📈 Visualization** — Interactive signal plots (Plotly) and static plots (Matplotlib)
- **🔧 Preprocessing** — Bandpass/notch filtering, re-referencing, ICA artifact removal, bad channel detection
- **📊 Analysis** — Power spectral density (PSD), frequency band power, channel connectivity
- **💾 Export** — Download processed data as CSV or FIF

## Quick Start

### Prerequisites

- Python 3.9+

### Installation

```bash
# Clone the repository
git clone https://github.com/wshark-mike/EEG_Analysis_Platform.git
cd EEG_Analysis_Platform

# Install dependencies
pip install -r requirements.txt
```

### Run the Application

```bash
streamlit run app.py
```

The app will open in your browser at `http://localhost:8501`.

## Project Structure

```
EEG_Analysis_Platform/
├── app.py                         # Main Streamlit entry point (home page)
├── requirements.txt               # Python dependencies
├── .streamlit/
│   └── config.toml                # Streamlit theme configuration
├── pages/
│   ├── 1_📂_Data_Upload.py       # Data loading page
│   ├── 2_📈_Visualization.py     # Signal visualization page
│   ├── 3_🔧_Preprocessing.py     # Preprocessing tools page
│   └── 4_📊_Analysis.py          # Analysis and export page
├── utils/
│   ├── __init__.py
│   ├── data_loader.py             # EEG file loading (EDF, BDF, FIF, CSV, SET)
│   ├── preprocessing.py           # Filtering, re-referencing, ICA
│   ├── analysis.py                # PSD, band power, connectivity, ERP, TFR
│   └── visualization.py           # Plotting helpers (Matplotlib + Plotly)
└── tests/
    ├── __init__.py
    ├── test_data_loader.py
    ├── test_preprocessing.py
    └── test_analysis.py
```

## Usage Guide

### 1. Upload Data

Navigate to the **Data Upload** page and upload an EEG file. Supported formats:

| Format | Extension | Description |
|--------|-----------|-------------|
| EDF    | `.edf`    | European Data Format |
| BDF    | `.bdf`    | BioSemi Data Format |
| FIF    | `.fif`    | MNE-Python native format |
| SET    | `.set`    | EEGLAB format |
| CSV    | `.csv`    | Comma-separated values (columns = channels, rows = samples) |

### 2. Visualize Signals

On the **Visualization** page, view raw EEG signals with adjustable time window, channel count, and interactive or static plot modes.

### 3. Preprocess Data

The **Preprocessing** page provides:

- **Bad channel detection** — Automatic detection using z-score analysis
- **Bandpass filter** — Set low and high frequency cutoffs
- **Notch filter** — Remove power line noise (50 Hz or 60 Hz with harmonics)
- **Re-referencing** — Average reference or single-channel reference
- **ICA** — Independent Component Analysis for artifact removal
- **Reset** — Restore original unprocessed data

### 4. Analyze

The **Analysis** page includes:

- **PSD** — Power Spectral Density using Welch or multitaper methods
- **Band Power** — Power in Delta, Theta, Alpha, Beta, and Gamma bands
- **Connectivity** — Channel correlation matrix
- **Export** — Download processed data as CSV or FIF

## Running Tests

```bash
python -m pytest tests/ -v
```

## Dependencies

- [Streamlit](https://streamlit.io/) — Web application framework
- [MNE-Python](https://mne.tools/) — EEG/MEG data processing
- [NumPy](https://numpy.org/) / [SciPy](https://scipy.org/) — Numerical computation
- [Matplotlib](https://matplotlib.org/) / [Plotly](https://plotly.com/) — Visualization
- [Pandas](https://pandas.pydata.org/) — Data manipulation
- [scikit-learn](https://scikit-learn.org/) — Required for ICA (FastICA)