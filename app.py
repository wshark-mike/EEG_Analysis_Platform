"""
EEG Analysis Platform - Main Application

Streamlit-based interactive platform for EEG data analysis, preprocessing,
visualization, and advanced signal processing.

Features:
- Multi-format EEG file loading (EDF, BDF, FIF, SET, CSV, BrainVision)
- Interactive signal visualization
- Advanced preprocessing (filtering, ICA, re-referencing)
- Frequency analysis (PSD, band power, connectivity)
- Session state management with undo functionality
- Memory optimization
"""

import streamlit as st
import numpy as np
import mne
from typing import Optional, List, Dict
import traceback

# Import utilities
from utils.data_loader import load_eeg_file, get_raw_info, get_supported_formats
from utils.preprocessing import (
    apply_bandpass_filter,
    apply_notch_filter,
    apply_rereferencing,
    run_ica,
    apply_ica_exclusion,
    detect_bad_channels,
)
from utils.visualization import (
    plot_raw_signals,
    plot_raw_signals_plotly,
)
from utils.analysis import (
    compute_psd,
    compute_band_power,
    compute_connectivity,
)
from utils.logger import get_logger, log_to_streamlit
from config import (
    BANDPASS_L_FREQ,
    BANDPASS_H_FREQ,
    NOTCH_FREQS_EU_ASIA,
    NOTCH_FREQS_AMERICAS,
    ICA_DEFAULT_N_COMPONENTS,
    BAD_CHANNEL_ZSCORE_THRESHOLD,
    FREQUENCY_BANDS,
)

# Initialize logger
logger = get_logger("app")

# ===== Streamlit Configuration =====
st.set_page_config(
    page_title="🧠 EEG Analysis Platform",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ===== CSS Styling =====
st.markdown("""
    <style>
    .main {
        padding-top: 2rem;
    }
    .stMetric {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
    }
    .success-box {
        background-color: #d4edda;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #28a745;
    }
    .error-box {
        background-color: #f8d7da;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #dc3545;
    }
    </style>
""", unsafe_allow_html=True)


# ===== Session State Initialization =====
def initialize_session_state():
    """Initialize all session state variables."""
    if "raw" not in st.session_state:
        st.session_state.raw = None
        logger.debug("Initialized session state: raw")

    if "raw_original" not in st.session_state:
        st.session_state.raw_original = None
        logger.debug("Initialized session state: raw_original")

    if "raw_history" not in st.session_state:
        st.session_state.raw_history = []
        logger.debug("Initialized session state: raw_history")

    if "pipeline" not in st.session_state:
        st.session_state.pipeline = []
        logger.debug("Initialized session state: pipeline")

    if "ica" not in st.session_state:
        st.session_state.ica = None
        logger.debug("Initialized session state: ica")

    if "psd_data" not in st.session_state:
        st.session_state.psd_data = None
        logger.debug("Initialized session state: psd_data")

    if "file_loaded" not in st.session_state:
        st.session_state.file_loaded = False
        logger.debug("Initialized session state: file_loaded")


# ===== Helper Functions =====
def add_processing_step(step_name: str) -> None:
    """
    Record a processing step in the pipeline.

    Parameters
    ----------
    step_name : str
        Name of the processing step.
    """
    st.session_state.pipeline.append(step_name)
    logger.debug(f"Added pipeline step: {step_name}")


def save_to_history() -> None:
    """Save current raw data to history for undo functionality."""
    if st.session_state.raw is not None:
        st.session_state.raw_history.append(st.session_state.raw.copy())
        logger.debug(f"Saved to history (depth: {len(st.session_state.raw_history)})")


def undo_last_step() -> None:
    """Undo the last processing step."""
    if len(st.session_state.raw_history) > 0:
        st.session_state.raw = st.session_state.raw_history.pop()
        st.session_state.pipeline.pop()
        logger.info("Undo: reverted to previous state")
        st.success("✅ Undo successful!")
        st.rerun()
    else:
        logger.warning("Undo attempted but history is empty")
        st.warning("⚠️ No history to undo")


def clear_cache_and_memory() -> None:
    """Clear cache and free memory."""
    st.session_state.raw = None
    st.session_state.raw_original = None
    st.session_state.raw_history = []
    st.session_state.ica = None
    st.session_state.psd_data = None
    st.session_state.pipeline = []
    st.session_state.file_loaded = False

    logger.info("Cleared session state and memory")
    st.success("✅ Cache and memory cleared!")


def format_duration(duration_sec: float) -> str:
    """
    Format duration in seconds to readable string.

    Parameters
    ----------
    duration_sec : float
        Duration in seconds.

    Returns
    -------
    str
        Formatted duration string (e.g., "1m 30s").
    """
    minutes = int(duration_sec) // 60
    seconds = int(duration_sec) % 60
    if minutes > 0:
        return f"{minutes}m {seconds}s"
    return f"{seconds}s"


# ===== Main Application =====
def main():
    """Main application function."""
    logger.info("=" * 80)
    logger.info("EEG Analysis Platform - Application Started")
    logger.info("=" * 80)

    # Initialize session state
    initialize_session_state()

    # ===== Main Title =====
    st.markdown("# 🧠 EEG Analysis Platform")
    st.markdown(
        "Interactive platform for EEG signal processing, analysis, and visualization"
    )

    # ===== Sidebar Configuration =====
    with st.sidebar:
        st.markdown("## 📊 Data Status")

        # Data loading status
        if st.session_state.file_loaded and st.session_state.raw is not None:
            raw = st.session_state.raw
            info = get_raw_info(raw)

            with st.container():
                col1, col2 = st.columns(2)
                with col1:
                    st.metric("📈 Channels", info["n_channels"])
                with col2:
                    st.metric("📡 Sample Rate", f"{info['sfreq']} Hz")

                col1, col2 = st.columns(2)
                with col1:
                    st.metric(
                        "⏱️ Duration",
                        format_duration(info["duration_sec"])
                    )
                with col2:
                    st.metric("📊 Samples", f"{info['n_samples']:,}")

            st.success("✅ Data Loaded Successfully")

            # Display channel names
            with st.expander("📋 Channel Names"):
                st.write(info["ch_names"])

            # Display channel types
            with st.expander("🔧 Channel Types"):
                for ch_name, ch_type in zip(info["ch_names"], info["ch_types"]):
                    st.write(f"  • {ch_name}: {ch_type}")

        else:
            st.info("ℹ️ No data loaded. Go to 'Data Upload' to load EEG files.")

        # ===== Processing Pipeline =====
        st.markdown("## 🛤️ Processing Pipeline")
        if st.session_state.pipeline:
            st.info(f"✅ {len(st.session_state.pipeline)} steps completed")
            with st.expander("View Pipeline"):
                for i, step in enumerate(st.session_state.pipeline, 1):
                    st.write(f"{i}. {step}")
        else:
            st.info("No preprocessing steps applied yet")

        # ===== Undo and Clear Buttons =====
        st.markdown("## 🔧 Tools")
        col1, col2 = st.columns(2)

        with col1:
            if st.button("↩️ Undo Last Step", use_container_width=True):
                undo_last_step()

        with col2:
            if st.button("🧹 Clear All", use_container_width=True):
                clear_cache_and_memory()
                st.rerun()

        # ===== Settings =====
        st.markdown("## ⚙️ Settings")
        st.markdown("### Filter Parameters")
        l_freq = st.number_input(
            "Low frequency (Hz)",
            value=BANDPASS_L_FREQ,
            min_value=0.0,
            max_value=100.0,
            step=0.1,
            help="Low cutoff frequency for bandpass filter",
        )
        h_freq = st.number_input(
            "High frequency (Hz)",
            value=BANDPASS_H_FREQ,
            min_value=0.0,
            max_value=500.0,
            step=1.0,
            help="High cutoff frequency for bandpass filter",
        )

        st.markdown("### Notch Filter")
        notch_freq = st.radio(
            "Power line frequency",
            options=[50.0, 60.0],
            horizontal=True,
            help="Select power line frequency for your region",
        )

        st.markdown("### ICA Parameters")
        n_components = st.slider(
            "Number of ICA components",
            min_value=5,
            max_value=30,
            value=ICA_DEFAULT_N_COMPONENTS,
            help="Number of independent components to extract",
        )

        st.markdown("### Bad Channel Detection")
        zscore_threshold = st.slider(
            "Z-score threshold",
            min_value=1.0,
            max_value=5.0,
            value=BAD_CHANNEL_ZSCORE_THRESHOLD,
            step=0.1,
            help="Higher value = stricter detection",
        )

    # ===== Main Content Area =====
    # Create tabs for different sections
    tab1, tab2, tab3, tab4, tab5 = st.tabs(
        [
            "📊 Overview",
            "📂 Data Upload",
            "📈 Visualization",
            "🔧 Preprocessing",
            "📊 Analysis",
        ]
    )

    # ===== Tab 1: Overview =====
    with tab1:
        st.markdown("## 📊 Platform Overview")

        col1, col2, col3 = st.columns(3)

        with col1:
            st.markdown("### ✨ Features")
            st.markdown(
                """
                - 🔄 Multi-format file loading
                - 📊 Real-time visualization
                - 🧠 Advanced preprocessing
                - 📈 Frequency analysis
                - 🔗 Connectivity analysis
                """
            )

        with col2:
            st.markdown("### 📁 Supported Formats")
            formats = get_supported_formats()
            for ext, name in formats.items():
                st.markdown(f"- **{ext}**: {name}")

        with col3:
            st.markdown("### 🎯 Quick Start")
            st.markdown(
                """
                1. Go to **Data Upload**
                2. Load your EEG file
                3. View in **Visualization**
                4. Apply filters in **Preprocessing**
                5. Analyze in **Analysis**
                """
            )

        st.markdown("---")

        st.markdown("### 📚 Frequency Bands")
        col1, col2, col3, col4, col5 = st.columns(5)

        bands = FREQUENCY_BANDS
        cols = [col1, col2, col3, col4, col5]

        for (band_name, (low_freq, high_freq)), col in zip(bands.items(), cols):
            with col:
                st.metric(band_name, f"{low_freq}-{high_freq} Hz")

    # ===== Tab 2: Data Upload =====
    with tab2:
        st.markdown("## 📂 Data Upload")

        uploaded_file = st.file_uploader(
            "Choose an EEG file",
            type=list(get_supported_formats().keys()),
            help="Select a supported EEG file format",
        )

        if uploaded_file is not None:
            try:
                logger.info(f"File uploaded: {uploaded_file.name}")

                # Save uploaded file temporarily
                import tempfile
                import os

                with tempfile.NamedTemporaryFile(
                    delete=False, suffix=os.path.splitext(uploaded_file.name)[1]
                ) as tmp_file:
                    tmp_file.write(uploaded_file.getbuffer())
                    tmp_file_path = tmp_file.name

                st.info("⏳ Loading file... (this may take a moment)")

                # Load EEG file
                raw = load_eeg_file(tmp_file_path)

                # Store in session state
                st.session_state.raw = raw
                st.session_state.raw_original = raw.copy()
                st.session_state.file_loaded = True
                st.session_state.pipeline = []
                st.session_state.raw_history = []

                logger.info(
                    f"File loaded successfully: {len(raw.ch_names)} channels, "
                    f"{raw.n_times / raw.info['sfreq']:.1f}s duration"
                )

                # Clean up temp file
                os.unlink(tmp_file_path)

                st.success("✅ File loaded successfully!")

                # Display file info
                info = get_raw_info(raw)
                col1, col2, col3, col4 = st.columns(4)

                with col1:
                    st.metric("Channels", info["n_channels"])
                with col2:
                    st.metric("Sample Rate", f"{info['sfreq']} Hz")
                with col3:
                    st.metric("Duration", format_duration(info["duration_sec"]))
                with col4:
                    st.metric("Samples", f"{info['n_samples']:,}")

                # Display channel names
                st.markdown("### 📋 Channels")
                st.write(", ".join(info["ch_names"]))

            except Exception as e:
                logger.error(f"Error loading file: {str(e)}\n{traceback.format_exc()}")
                st.error(f"❌ Error loading file: {str(e)}")
                log_to_streamlit("error", f"File loading failed: {str(e)}", show_in_ui=False)

        else:
            st.info("👆 Upload an EEG file to get started")

    # ===== Tab 3: Visualization =====
    with tab3:
        st.markdown("## 📈 Visualization")

        if not st.session_state.file_loaded or st.session_state.raw is None:
            st.warning("⚠️ Please load data first (go to 'Data Upload')")
        else:
            try:
                raw = st.session_state.raw
                info = get_raw_info(raw)

                # Visualization options
                col1, col2, col3 = st.columns(3)

                with col1:
                    duration = st.slider(
                        "Duration (seconds)",
                        min_value=1.0,
                        max_value=min(60.0, info["duration_sec"]),
                        value=min(10.0, info["duration_sec"]),
                        step=1.0,
                    )

                with col2:
                    start_time = st.slider(
                        "Start time (seconds)",
                        min_value=0.0,
                        max_value=max(0.0, info["duration_sec"] - duration),
                        step=1.0,
                    )

                with col3:
                    n_channels = st.number_input(
                        "Number of channels to plot",
                        min_value=1,
                        max_value=info["n_channels"],
                        value=min(16, info["n_channels"]),
                    )

                # Visualization type
                viz_type = st.radio(
                    "Visualization type",
                    options=["Interactive (Plotly)", "Static (Matplotlib)"],
                    horizontal=True,
                )

                st.markdown("---")

                try:
                    logger.info(
                        f"Creating visualization: duration={duration}s, "
                        f"start={start_time}s, channels={n_channels}"
                    )

                    if viz_type == "Interactive (Plotly)":
                        fig = plot_raw_signals_plotly(
                            raw,
                            duration=duration,
                            start=start_time,
                            n_channels=n_channels,
                        )
                    else:
                        fig = plot_raw_signals(
                            raw,
                            duration=duration,
                            start=start_time,
                            n_channels=n_channels,
                        )

                    st.plotly_chart(fig, use_container_width=True)
                    logger.debug("Visualization created successfully")

                except Exception as e:
                    logger.error(f"Visualization error: {str(e)}\n{traceback.format_exc()}")
                    st.error(f"❌ Visualization error: {str(e)}")

            except Exception as e:
                logger.error(f"Tab 3 error: {str(e)}\n{traceback.format_exc()}")
                st.error(f"❌ Error: {str(e)}")

    # ===== Tab 4: Preprocessing =====
    with tab4:
        st.markdown("## 🔧 Preprocessing")

        if not st.session_state.file_loaded or st.session_state.raw is None:
            st.warning("⚠️ Please load data first (go to 'Data Upload')")
        else:
            try:
                raw = st.session_state.raw

                st.markdown("### 🎯 Select Preprocessing Steps")

                # Preprocessing options
                col1, col2 = st.columns(2)

                with col1:
                    apply_bandpass = st.checkbox(
                        "Apply Bandpass Filter",
                        help="Filter between specified frequencies",
                    )
                    apply_notch = st.checkbox(
                        "Apply Notch Filter",
                        help="Remove power line noise (50/60 Hz)",
                    )
                    apply_reref = st.checkbox(
                        "Apply Re-referencing",
                        help="Change reference electrode",
                    )
                    detect_bad = st.checkbox(
                        "Detect Bad Channels",
                        help="Identify noisy channels",
                    )

                with col2:
                    run_ica_check = st.checkbox(
                        "Run ICA",
                        help="Independent Component Analysis for artifact removal",
                    )

                st.markdown("---")

                # Apply preprocessing
                if st.button("▶️ Apply Preprocessing", use_container_width=True):
                    try:
                        save_to_history()
                        modified = False

                        # Bandpass filter
                        if apply_bandpass:
                            logger.info(
                                f"Applying bandpass filter: {l_freq}-{h_freq} Hz"
                            )
                            st.info("⏳ Applying bandpass filter...")
                            raw = apply_bandpass_filter(raw, l_freq=l_freq, h_freq=h_freq)
                            add_processing_step(
                                f"Bandpass Filter ({l_freq}-{h_freq} Hz)"
                            )
                            modified = True
                            st.success("✅ Bandpass filter applied")

                        # Notch filter
                        if apply_notch:
                            logger.info(f"Applying notch filter at {notch_freq} Hz")
                            st.info("⏳ Applying notch filter...")
                            raw = apply_notch_filter(raw, freqs=notch_freq)
                            add_processing_step(f"Notch Filter ({notch_freq} Hz)")
                            modified = True
                            st.success("✅ Notch filter applied")

                        # Re-referencing
                        if apply_reref:
                            logger.info("Applying re-referencing")
                            st.info("⏳ Applying re-referencing...")
                            raw = apply_rereferencing(raw, ref_type="average")
                            add_processing_step("Re-referencing (Average)")
                            modified = True
                            st.success("✅ Re-referencing applied")

                        # Bad channel detection
                        if detect_bad:
                            logger.info("Detecting bad channels")
                            st.info("⏳ Detecting bad channels...")
                            bad_channels = detect_bad_channels(
                                raw,
                                threshold=zscore_threshold
                            )
                            if bad_channels:
                                st.warning(
                                    f"⚠️ Detected bad channels: {', '.join(bad_channels)}"
                                )
                                raw.info["bads"] = bad_channels
                                add_processing_step(
                                    f"Bad Channel Detection ({len(bad_channels)} found)"
                                )
                                modified = True
                                logger.warning(f"Bad channels detected: {bad_channels}")
                            else:
                                st.success("✅ No bad channels detected")

                        # ICA
                        if run_ica_check:
                            logger.info(f"Running ICA with {n_components} components")
                            st.info("⏳ Running ICA (this may take a moment)...")
                            ica = run_ica(raw, n_components=n_components)
                            st.session_state.ica = ica
                            add_processing_step(f"ICA ({n_components} components)")
                            modified = True
                            st.success("✅ ICA completed")
                            logger.info("ICA fitting completed")

                        # Update session state
                        if modified:
                            st.session_state.raw = raw
                            logger.info("Preprocessing completed and session state updated")
                            st.success("✅ All preprocessing steps completed!")

                    except Exception as e:
                        logger.error(
                            f"Preprocessing error: {str(e)}\n{traceback.format_exc()}"
                        )
                        st.error(f"❌ Preprocessing error: {str(e)}")
                        # Restore from history on error
                        if st.session_state.raw_history:
                            st.session_state.raw = st.session_state.raw_history.pop()

            except Exception as e:
                logger.error(f"Tab 4 error: {str(e)}\n{traceback.format_exc()}")
                st.error(f"❌ Error: {str(e)}")

    # ===== Tab 5: Analysis =====
    with tab5:
        st.markdown("## 📊 Analysis")

        if not st.session_state.file_loaded or st.session_state.raw is None:
            st.warning("⚠️ Please load data first (go to 'Data Upload')")
        else:
            try:
                raw = st.session_state.raw

                st.markdown("### 🎯 Select Analysis Type")

                analysis_type = st.radio(
                    "Analysis type",
                    options=["PSD Analysis", "Band Power", "Connectivity"],
                    horizontal=True,
                )

                st.markdown("---")

                if analysis_type == "PSD Analysis":
                    st.markdown("### Power Spectral Density (PSD)")

                    if st.button("📈 Compute PSD", use_container_width=True):
                        try:
                            logger.info("Computing PSD")
                            st.info("⏳ Computing PSD...")
                            psd, freqs = compute_psd(raw)
                            st.session_state.psd_data = (psd, freqs)
                            st.success("✅ PSD computed")
                            logger.debug("PSD computation completed")

                            # Plot PSD
                            from utils.visualization import plot_psd
                            fig = plot_psd(psd, freqs, raw.ch_names, log_scale=True)
                            st.plotly_chart(fig, use_container_width=True)

                        except Exception as e:
                            logger.error(
                                f"PSD error: {str(e)}\n{traceback.format_exc()}"
                            )
                            st.error(f"❌ PSD computation error: {str(e)}")

                elif analysis_type == "Band Power":
                    st.markdown("### Band Power Analysis")

                    if st.button("📊 Compute Band Power", use_container_width=True):
                        try:
                            logger.info("Computing band power")
                            st.info("⏳ Computing band power...")
                            band_powers = compute_band_power(raw)
                            st.success("✅ Band power computed")
                            logger.debug("Band power computation completed")

                            # Plot band power
                            from utils.visualization import plot_band_power
                            fig = plot_band_power(band_powers, raw.ch_names)
                            st.plotly_chart(fig, use_container_width=True)

                        except Exception as e:
                            logger.error(
                                f"Band power error: {str(e)}\n{traceback.format_exc()}"
                            )
                            st.error(f"❌ Band power computation error: {str(e)}")

                elif analysis_type == "Connectivity":
                    st.markdown("### Connectivity Analysis")

                    if st.button("🔗 Compute Connectivity", use_container_width=True):
                        try:
                            logger.info("Computing connectivity")
                            st.info("⏳ Computing connectivity (this may take a moment)...")
                            conn_matrix = compute_connectivity(raw)
                            st.success("✅ Connectivity computed")
                            logger.debug("Connectivity computation completed")

                            # Plot connectivity
                            from utils.visualization import plot_connectivity_matrix
                            fig = plot_connectivity_matrix(conn_matrix, raw.ch_names)
                            st.plotly_chart(fig, use_container_width=True)

                        except Exception as e:
                            logger.error(
                                f"Connectivity error: {str(e)}\n{traceback.format_exc()}"
                            )
                            st.error(f"❌ Connectivity computation error: {str(e)}")

            except Exception as e:
                logger.error(f"Tab 5 error: {str(e)}\n{traceback.format_exc()}")
                st.error(f"❌ Error: {str(e)}")

    # ===== Footer =====
    st.markdown("---")
    st.markdown(
        """
        <div style="text-align: center; color: #888; font-size: 0.9em;">
        <p>🧠 EEG Analysis Platform | Built with Streamlit & MNE-Python</p>
        <p>For issues or feature requests, visit: <a href="https://github.com/wshark-mike/EEG_Analysis_Platform">GitHub</a></p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    logger.info("Application render completed successfully")


# ===== Run Application =====
if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        logger.critical(f"Critical error in main: {str(e)}\n{traceback.format_exc()}")
        st.error(f"❌ Critical Error: {str(e)}")