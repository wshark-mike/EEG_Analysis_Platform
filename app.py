"""
EEG Analysis Platform - Main Application

A Streamlit-based platform for EEG data analysis, providing tools for
data loading, visualization, preprocessing, and frequency/time analysis.
"""

import streamlit as st

st.set_page_config(
    page_title="EEG Analysis Platform",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)


def main():
    st.title("🧠 EEG Analysis Platform")
    st.markdown("---")

    st.markdown(
        """
        Welcome to the **EEG Analysis Platform**! This tool helps you analyze
        electroencephalography (EEG) data through an intuitive web interface.

        ### Features

        - **📂 Data Upload** — Load EEG data from EDF, BDF, FIF, SET, or CSV files
        - **📈 Visualization** — View raw signals with interactive plots
        - **🔧 Preprocessing** — Apply filters, re-referencing, and ICA artifact removal
        - **📊 Analysis** — Compute PSD, band power, and channel connectivity

        ### Getting Started

        1. Navigate to the **Data Upload** page from the sidebar
        2. Upload your EEG data file
        3. Explore, preprocess, and analyze your data

        ### Supported File Formats

        | Format | Extension | Description |
        |--------|-----------|-------------|
        | EDF    | `.edf`    | European Data Format |
        | BDF    | `.bdf`    | BioSemi Data Format |
        | FIF    | `.fif`    | MNE-Python native format |
        | SET    | `.set`    | EEGLAB format |
        | CSV    | `.csv`    | Comma-Separated Values |
        """
    )

    # Show current data status in sidebar
    st.sidebar.markdown("### 📋 Data Status")
    if "raw" in st.session_state and st.session_state.raw is not None:
        raw = st.session_state.raw
        st.sidebar.success("✅ Data loaded")
        st.sidebar.info(
            f"**Channels:** {len(raw.ch_names)}  \n"
            f"**Sample Rate:** {raw.info['sfreq']} Hz  \n"
            f"**Duration:** {raw.n_times / raw.info['sfreq']:.1f} s"
        )
    else:
        st.sidebar.warning("⚠️ No data loaded. Go to **Data Upload** to begin.")


if __name__ == "__main__":
    main()
