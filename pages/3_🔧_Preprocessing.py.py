"""
EEG Preprocessing Page

Interactive preprocessing interface for applying filters, ICA, bad channel detection, etc.
"""

import streamlit as st
import traceback
from typing import Optional

import mne

from utils.preprocessing import (
    apply_bandpass_filter,
    apply_notch_filter,
    apply_rereferencing,
    run_ica,
    apply_ica_exclusion,
    detect_bad_channels,
)
from utils.data_loader import get_raw_info
from utils.visualization import (
    plot_raw_signals,
    plot_raw_signals_plotly,
    plot_ica_components,
)
from utils.logger import get_logger, log_to_streamlit
from config import (
    BANDPASS_L_FREQ,
    BANDPASS_H_FREQ,
    NOTCH_FREQS_EU_ASIA,
    ICA_DEFAULT_N_COMPONENTS,
    BAD_CHANNEL_ZSCORE_THRESHOLD,
    MAX_HISTORY_DEPTH,
)

logger = get_logger("preprocessing_page")

st.set_page_config(page_title="🔧 Preprocessing", page_icon="🔧", layout="wide")

st.markdown("# 🔧 EEG Preprocessing")
st.markdown("Apply various preprocessing steps to your EEG data")


# ===== Initialize Session State =====
def initialize_session_state():
    """Initialize all required session state variables."""
    if "raw" not in st.session_state:
        st.session_state.raw = None
    if "raw_original" not in st.session_state:
        st.session_state.raw_original = None
    if "raw_history" not in st.session_state:
        st.session_state.raw_history = []
    if "file_loaded" not in st.session_state:
        st.session_state.file_loaded = False
    if "pipeline" not in st.session_state:
        st.session_state.pipeline = []
    if "ica" not in st.session_state:
        st.session_state.ica = None


initialize_session_state()

# ===== Check if data is loaded =====
# ✅ 更灵活的检查 - 只要 raw 不为 None 就认为数据已加载
if st.session_state.raw is None:
    st.error("❌ No data loaded!")
    st.info("📂 **How to load data:**")
    st.markdown(
        """
    1. Click on the **"Data Upload"** tab at the top
    2. Upload your EEG file (.edf, .bdf, .fif, .set, .csv, .vhdr)
    3. Wait for the file to load
    4. Then come back to this page
    """
    )

    # Add debug info
    with st.expander("🔧 Debug Info"):
        st.write(f"file_loaded: {st.session_state.get('file_loaded', 'NOT SET')}")
        st.write(f"raw: {st.session_state.raw}")
        st.write(f"Session state keys: {list(st.session_state.keys())}")

    st.stop()

try:
    raw = st.session_state.raw
    info = get_raw_info(raw)

    # ===== Sidebar Settings =====
    with st.sidebar:
        st.markdown("## ⚙️ Settings")

        st.markdown("### 🎚️ Filter Parameters")
        l_freq = st.number_input(
            "Low frequency (Hz)",
            value=BANDPASS_L_FREQ,
            min_value=0.0,
            max_value=100.0,
            step=0.1,
            help="Low cutoff frequency for bandpass filter",
            key="l_freq_input",
        )

        h_freq = st.number_input(
            "High frequency (Hz)",
            value=BANDPASS_H_FREQ,
            min_value=1.0,
            max_value=500.0,
            step=1.0,
            help="High cutoff frequency for bandpass filter",
            key="h_freq_input",
        )

        st.markdown("### 🔌 Notch Filter")
        notch_freq = st.radio(
            "Power line frequency",
            options=[50.0, 60.0],
            horizontal=True,
            help="Select your region's power line frequency",
            key="notch_freq_input",
        )

        st.markdown("### 🧠 ICA Parameters")
        n_components = st.slider(
            "Number of ICA components",
            min_value=5,
            max_value=min(30, info["n_channels"]),
            value=min(ICA_DEFAULT_N_COMPONENTS, info["n_channels"]),
            help="Number of independent components to extract",
            key="n_components_input",
        )

        st.markdown("### 🚨 Bad Channel Detection")
        zscore_threshold = st.slider(
            "Z-score threshold",
            min_value=1.0,
            max_value=5.0,
            value=BAD_CHANNEL_ZSCORE_THRESHOLD,
            step=0.1,
            help="Higher = stricter detection",
            key="zscore_threshold_input",
        )

    # ===== Main Content =====
    st.markdown("## 📊 Current Data Status")

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("📈 Channels", info["n_channels"])
    with col2:
        st.metric("📡 Sample Rate", f"{info['sfreq']} Hz")
    with col3:
        st.metric("⏱️ Duration", f"{info['duration_sec']:.1f}s")
    with col4:
        st.metric("📊 Samples", f"{info['n_samples']:,}")

    st.success(f"✅ Data loaded: {len(info['ch_names'])} channels")

    # ===== Processing Pipeline Display =====
    st.markdown("## 🛤️ Processing Pipeline")

    if st.session_state.get("pipeline"):
        st.info(f"✅ {len(st.session_state.pipeline)} steps completed")
        with st.expander("View Pipeline Details"):
            for i, step in enumerate(st.session_state.pipeline, 1):
                st.write(f"  {i}. ✅ {step}")
    else:
        st.info("No preprocessing steps applied yet. Select steps below to begin.")

    # ===== ICA Status =====
    if st.session_state.get("ica") is not None:
        st.success(f"✅ ICA fitted ({st.session_state.ica.n_components} components)")
    else:
        st.info("ℹ️ No ICA fitted yet. Run ICA in the preprocessing steps below.")

    # ===== Preprocessing Steps Selection =====
    st.markdown("## 🎯 Select Preprocessing Steps")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("### Filter Steps")
        apply_bandpass = st.checkbox(
            "✓ Apply Bandpass Filter",
            value=False,
            help="Filter between specified frequencies to remove very low and high frequency noise",
            key="apply_bandpass_cb",
        )

        apply_notch = st.checkbox(
            "✓ Apply Notch Filter",
            value=False,
            help="Remove power line noise (50/60 Hz)",
            key="apply_notch_cb",
        )

        apply_reref = st.checkbox(
            "✓ Apply Re-referencing",
            value=False,
            help="Change the reference electrode (common reference is average)",
            key="apply_reref_cb",
        )

    with col2:
        st.markdown("### Artifact Removal")
        detect_bad = st.checkbox(
            "✓ Detect Bad Channels",
            value=False,
            help="Identify noisy channels based on statistical analysis",
            key="detect_bad_cb",
        )

        run_ica_check = st.checkbox(
            "✓ Run ICA",
            value=False,
            help="Independent Component Analysis for artifact removal",
            key="run_ica_cb",
        )

    st.markdown("---")

    # ===== Apply Preprocessing Button =====
    col1, col2, col3 = st.columns([2, 1, 1])

    with col1:
        apply_button = st.button(
            "▶️ Apply Preprocessing",
            use_container_width=True,
            key="apply_preprocessing_btn",
        )

    with col2:
        preview_button = st.button(
            "👁️ Preview",
            use_container_width=True,
            key="preview_btn",
        )

    with col3:
        reset_button = st.button(
            "🔄 Reset",
            use_container_width=True,
            key="reset_btn",
        )

    # ===== Reset Button Handler =====
    if reset_button:
        logger.info("User clicked reset button")
        try:
            if st.session_state.raw_original is not None:
                st.session_state.raw = st.session_state.raw_original.copy()
                st.session_state.pipeline = []
                st.session_state.raw_history = []
                st.session_state.ica = None
                logger.info("Data reset to original state")
                st.success("✅ Data reset to original state")
                st.rerun()
            else:
                st.warning("⚠️ Cannot reset: original data not found")
        except Exception as e:
            logger.error(f"Reset error: {str(e)}")
            st.error(f"❌ Reset error: {str(e)}")

    # ===== Preview Button Handler =====
    if preview_button:
        logger.info("User clicked preview button")
        st.markdown("### 👁️ Signal Preview (Current Data)")

        try:
            duration = min(10.0, info["duration_sec"])
            fig = plot_raw_signals_plotly(
                st.session_state.raw,
                duration=duration,
                n_channels=min(16, info["n_channels"]),
            )
            st.plotly_chart(fig, use_container_width=True)
            logger.debug("Preview displayed successfully")
        except Exception as e:
            logger.error(f"Preview error: {str(e)}\n{traceback.format_exc()}")
            st.error(f"❌ Preview error: {str(e)}")

    # ===== Apply Preprocessing Button Handler =====
    if apply_button:
        logger.info("User clicked apply preprocessing button")
        logger.info(
            f"Selected steps: bandpass={apply_bandpass}, notch={apply_notch}, "
            f"reref={apply_reref}, bad_ch={detect_bad}, ica={run_ica_check}"
        )

        # Check if at least one step is selected
        if not any(
            [apply_bandpass, apply_notch, apply_reref, detect_bad, run_ica_check]
        ):
            st.warning("⚠️ Please select at least one preprocessing step")
            logger.warning("User clicked apply but no steps selected")
        else:
            # Create a progress container
            progress_container = st.container()

            with progress_container:
                st.markdown("### ⏳ Processing...")
                status_placeholder = st.empty()

                try:
                    # 🆕 Enforce history depth limit to prevent memory leaks
                    if len(st.session_state.raw_history) >= MAX_HISTORY_DEPTH:
                        removed = st.session_state.raw_history.pop(0)
                        logger.info(
                            f"History depth limit ({MAX_HISTORY_DEPTH}) reached. "
                            f"Removed oldest entry."
                        )
                    
                    # Save current state to history for undo
                    st.session_state.raw_history.append(st.session_state.raw.copy())
                    logger.debug(
                        f"Saved to history ({len(st.session_state.raw_history)}/"
                        f"{MAX_HISTORY_DEPTH})"
                    )

                    current_raw = st.session_state.raw.copy()
                    step_count = 0
                    total_steps = sum(
                        [
                            apply_bandpass,
                            apply_notch,
                            apply_reref,
                            detect_bad,
                            run_ica_check,
                        ]
                    )

                    # ===== Step 1: Bandpass Filter =====
                    if apply_bandpass:
                        step_count += 1
                        status_placeholder.empty()
                        with status_placeholder.container():
                            st.info(
                                f"[{step_count}/{total_steps}] Applying bandpass filter..."
                            )

                        try:
                            logger.info(
                                f"Applying bandpass filter: {l_freq}-{h_freq} Hz"
                            )
                            current_raw = apply_bandpass_filter(
                                current_raw,
                                l_freq=l_freq,
                                h_freq=h_freq,
                            )
                            st.session_state.pipeline.append(
                                f"Bandpass Filter ({l_freq}-{h_freq} Hz)"
                            )
                            logger.debug("Bandpass filter applied successfully")

                        except Exception as e:
                            logger.error(
                                f"Bandpass filter error: {str(e)}\n{traceback.format_exc()}"
                            )
                            st.error(f"❌ Bandpass filter error: {str(e)}")
                            raise

                    # ===== Step 2: Notch Filter =====
                    if apply_notch:
                        step_count += 1
                        status_placeholder.empty()
                        with status_placeholder.container():
                            st.info(
                                f"[{step_count}/{total_steps}] Applying notch filter..."
                            )

                        try:
                            logger.info(f"Applying notch filter at {notch_freq} Hz")
                            current_raw = apply_notch_filter(
                                current_raw, freqs=notch_freq
                            )
                            st.session_state.pipeline.append(
                                f"Notch Filter ({notch_freq} Hz)"
                            )
                            logger.debug("Notch filter applied successfully")

                        except Exception as e:
                            logger.error(
                                f"Notch filter error: {str(e)}\n{traceback.format_exc()}"
                            )
                            st.error(f"❌ Notch filter error: {str(e)}")
                            raise

                    # ===== Step 3: Re-referencing =====
                    if apply_reref:
                        step_count += 1
                        status_placeholder.empty()
                        with status_placeholder.container():
                            st.info(
                                f"[{step_count}/{total_steps}] Applying re-referencing..."
                            )

                        try:
                            logger.info("Applying re-referencing (average)")
                            current_raw = apply_rereferencing(
                                current_raw, ref_type="average"
                            )
                            st.session_state.pipeline.append("Re-referencing (Average)")
                            logger.debug("Re-referencing applied successfully")

                        except Exception as e:
                            logger.error(
                                f"Re-referencing error: {str(e)}\n{traceback.format_exc()}"
                            )
                            st.error(f"❌ Re-referencing error: {str(e)}")
                            raise

                    # ===== Step 4: Bad Channel Detection =====
                    if detect_bad:
                        step_count += 1
                        status_placeholder.empty()
                        with status_placeholder.container():
                            st.info(
                                f"[{step_count}/{total_steps}] Detecting bad channels..."
                            )

                        try:
                            logger.info("Detecting bad channels")
                            bad_channels = detect_bad_channels(
                                current_raw,
                                threshold=zscore_threshold,
                            )

                            if bad_channels:
                                st.warning(
                                    f"⚠️ Detected {len(bad_channels)} bad channels: {', '.join(bad_channels)}"
                                )
                                current_raw.info["bads"] = bad_channels
                                st.session_state.pipeline.append(
                                    f"Bad Channel Detection ({len(bad_channels)} found)"
                                )
                                logger.warning(f"Bad channels detected: {bad_channels}")
                            else:
                                st.success("✅ No bad channels detected")
                                st.session_state.pipeline.append(
                                    "Bad Channel Detection (0 found)"
                                )

                        except Exception as e:
                            logger.error(
                                f"Bad channel detection error: {str(e)}\n{traceback.format_exc()}"
                            )
                            st.error(f"❌ Bad channel detection error: {str(e)}")
                            raise

                    # ===== Step 5: ICA =====
                    if run_ica_check:
                        step_count += 1
                        status_placeholder.empty()
                        with status_placeholder.container():
                            st.info(
                                f"[{step_count}/{total_steps}] Running ICA (this may take a moment)..."
                            )

                        try:
                            logger.info(f"Running ICA with {n_components} components")
                            ica = run_ica(current_raw, n_components=n_components)
                            st.session_state.ica = ica
                            st.session_state.pipeline.append(
                                f"ICA ({n_components} components)"
                            )
                            logger.info("ICA fitting completed successfully")

                        except Exception as e:
                            logger.error(
                                f"ICA error: {str(e)}\n{traceback.format_exc()}"
                            )
                            st.error(f"❌ ICA error: {str(e)}")
                            raise

                    # ===== Update Session State =====
                    st.session_state.raw = current_raw
                    logger.info("Session state updated with preprocessed data")

                    # Clear placeholder and show success
                    status_placeholder.empty()

                    st.success("✅ Preprocessing completed successfully!")
                    logger.info("All preprocessing steps completed successfully")

                    # Show summary
                    with st.expander("📋 Processing Summary", expanded=True):
                        st.markdown("#### Applied Steps:")
                        for i, step in enumerate(st.session_state.pipeline, 1):
                            st.write(f"  {i}. ✅ {step}")

                    # Force page rerun after a brief pause
                    import time

                    time.sleep(1)
                    st.rerun()

                except Exception as e:
                    # Restore from history on error
                    if st.session_state.raw_history:
                        st.session_state.raw = st.session_state.raw_history.pop()
                        if st.session_state.pipeline:
                            st.session_state.pipeline.pop()
                        logger.warning("Error occurred, restored from history")

                    status_placeholder.empty()

                    logger.error(
                        f"Preprocessing failed: {str(e)}\n{traceback.format_exc()}"
                    )
                    st.error(f"❌ Preprocessing failed: {str(e)}")
                    st.error("💡 Your data has been restored to the previous state")

    # ===== ICA Visualization Section =====
    if st.session_state.get("ica") is not None:
        st.markdown("## 🧠 ICA Visualization")
        st.markdown("Visualize and manage ICA components")

        ica_viz_col1, ica_viz_col2 = st.columns([2, 1])

        with ica_viz_col1:
            n_display = st.slider(
                "Number of components to display",
                min_value=1,
                max_value=st.session_state.ica.n_components,
                value=min(10, st.session_state.ica.n_components),
                key="n_display_ica",
            )

        with ica_viz_col2:
            if st.button(
                "📊 Plot Components", use_container_width=True, key="plot_ica_btn"
            ):
                try:
                    logger.info(f"Plotting {n_display} ICA components")
                    fig = plot_ica_components(
                        st.session_state.ica,
                        st.session_state.raw,
                        n_components=n_display,
                    )
                    st.pyplot(fig, use_container_width=True)
                    logger.debug("ICA components plotted successfully")
                except Exception as e:
                    logger.error(f"ICA plot error: {str(e)}\n{traceback.format_exc()}")
                    st.error(f"❌ ICA plot error: {str(e)}")

        # ===== ICA Component Exclusion =====
        st.markdown("### 🔄 Exclude Components")
        st.markdown("Select ICA components to exclude (remove artifacts)")

        exclude_components = st.multiselect(
            "Components to exclude",
            options=list(range(st.session_state.ica.n_components)),
            default=[],
            help="Select components that represent artifacts",
            key="exclude_components_select",
        )

        if st.button(
            "✓ Apply ICA Exclusion",
            use_container_width=True,
            key="apply_ica_exclusion_btn",
        ):
            if exclude_components:
                try:
                    logger.info(f"Excluding ICA components: {exclude_components}")
                    st.info(
                        f"⏳ Applying ICA exclusion for components: {exclude_components}..."
                    )

                    # Save to history
                    st.session_state.raw_history.append(st.session_state.raw.copy())

                    # Apply exclusion
                    raw_clean = apply_ica_exclusion(
                        st.session_state.raw, st.session_state.ica, exclude_components
                    )

                    # Update session state
                    st.session_state.raw = raw_clean
                    st.session_state.pipeline.append(
                        f"ICA Exclusion ({len(exclude_components)} components)"
                    )

                    logger.info(f"ICA exclusion applied successfully")
                    st.success(f"✅ Excluded {len(exclude_components)} ICA components!")
                    st.rerun()

                except Exception as e:
                    logger.error(
                        f"ICA exclusion error: {str(e)}\n{traceback.format_exc()}"
                    )
                    st.error(f"❌ ICA exclusion error: {str(e)}")
            else:
                st.warning("⚠️ Please select at least one component to exclude")

except Exception as e:
    logger.error(f"Page error: {str(e)}\n{traceback.format_exc()}")
    st.error(f"❌ Error: {str(e)}")
    st.error("💡 Please reload the page or go back to 'Data Upload'")

    with st.expander("🔧 Debug Info"):
        st.write(f"Error type: {type(e).__name__}")
        st.write(f"Error message: {str(e)}")
        st.write(traceback.format_exc())
