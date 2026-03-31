"""
Data Upload Page — Load EEG data files into the platform.
"""

import os
import shutil
import streamlit as st
from utils.data_loader import load_eeg_file, get_raw_info, get_supported_formats
from utils.logger import get_logger
from config import MAX_FILE_SIZE_MB, MAX_HISTORY_DEPTH

logger = get_logger("data_upload_page")

DATA_DIR = "data"
os.makedirs(DATA_DIR, exist_ok=True)

MAX_FILE_SIZE_BYTES = MAX_FILE_SIZE_MB * 1024 * 1024

st.set_page_config(page_title="Data Upload", page_icon="📂", layout="wide")

st.title("📂 Data Upload")
st.markdown("Upload your EEG data file to begin analysis.")

# Show supported formats
formats = get_supported_formats()
with st.expander("📋 Supported file formats"):
    for ext, desc in formats.items():
        st.markdown(f"- **{ext}** — {desc}")


def _display_raw_info(raw, filename):
    """
    Store loaded data in session state and display info metrics.
    Clears all stale analysis results from previous files.
    """
    # Load new data
    st.session_state.raw = raw
    st.session_state.raw_original = raw.copy()
    st.session_state.filename = filename
    st.session_state.file_loaded = True
    st.session_state.pipeline = [f"📂 Original data ({filename})"]
    st.session_state.raw_history = []

    # 🆕 Clear all stale analysis results
    st.session_state.ica = None
    st.session_state.psd_data = None
    st.session_state.psd_freqs = None
    st.session_state.psd_fig = None
    st.session_state.band_powers = None
    st.session_state.band_power_fig = None
    st.session_state.conn_matrix = None
    st.session_state.conn_ch_names = None
    st.session_state.conn_fig = None
    st.session_state.report_content = None

    # Clear visualization cache
    st.session_state.viz_fig_main = None
    st.session_state.viz_fig_single = None
    st.session_state.viz_type = None

    logger.info(
        f"Data loaded: {filename}, {len(raw.ch_names)} channels, "
        f"{raw.n_times} samples. All analysis results cleared."
    )

    st.success("✅ Data loaded successfully! All previous analysis results cleared.")

    info = get_raw_info(raw)
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("📈 Channels", info["n_channels"])
    col2.metric("📡 Sample Rate", f"{info['sfreq']} Hz")
    col3.metric("⏱️ Duration", f"{info['duration_sec']:.1f} s")
    col4.metric("📊 Samples", f"{info['n_samples']:,}")

    st.markdown("#### 📋 Channel Information")
    ch_data = {
        "Channel Name": info["ch_names"],
        "Type": info["ch_types"],
    }
    st.dataframe(ch_data, use_container_width=True)

    st.info(
        "✅ You can now proceed to the **Preprocessing** page to apply filters and analysis!"
    )


# --- Upload mode selector ---
upload_mode = st.radio(
    "Select upload mode",
    ["Single file (EDF / BDF / FIF / SET / CSV)", "BrainVision (.vhdr + .eeg + .vmrk)"],
    horizontal=True,
    key="upload_mode_radio",
)

if upload_mode == "Single file (EDF / BDF / FIF / SET / CSV)":
    # --- Single-file uploader ---
    uploaded_file = st.file_uploader(
        "Choose an EEG data file",
        type=["edf", "bdf", "fif", "set", "csv"],
        help="Upload EEG data in one of the supported formats.",
        key="single_file_uploader",
    )

    if uploaded_file is not None:
        # 🆕 File size validation
        file_size_mb = len(uploaded_file.getbuffer()) / 1024 / 1024
        if file_size_mb > MAX_FILE_SIZE_MB:
            st.error(
                f"❌ **File too large!**\n\n"
                f"- **Max size**: {MAX_FILE_SIZE_MB} MB\n"
                f"- **Your file**: {file_size_mb:.1f} MB\n\n"
                f"Please reduce file size or split into chunks."
            )
            st.stop()
        
        st.info(
            f"📄 File: **{uploaded_file.name}** ({file_size_mb:.1f} MB)"
        )

        basename = os.path.splitext(uploaded_file.name)[0]
        suffix = os.path.splitext(uploaded_file.name)[1].lower()

        sub_dir = os.path.join(DATA_DIR, basename)
        os.makedirs(sub_dir, exist_ok=True)

        with st.spinner("⏳ Loading EEG data..."):
            try:
                save_path = os.path.join(sub_dir, uploaded_file.name)
                with open(save_path, "wb") as f:
                    f.write(uploaded_file.getbuffer())

                logger.info(f"File saved to: {save_path}")

                raw = load_eeg_file(save_path, file_type=suffix)
                logger.info(f"File loaded successfully: {suffix} format")

                _display_raw_info(raw, uploaded_file.name)

                st.toast(f"✅ Saved to data/{basename}/", icon="💾")

            except Exception as e:
                logger.error(f"Failed to load file: {str(e)}")
                st.error(f"❌ Failed to load file: {e}")

else:
    # --- BrainVision multi-file uploader ---
    st.markdown(
        "Upload all three BrainVision files together: "
        "**.vhdr** (header), **.eeg** (data), **.vmrk** (markers)."
    )
    uploaded_files = st.file_uploader(
        "Choose BrainVision files (.vhdr, .eeg, .vmrk)",
        type=["vhdr", "eeg", "vmrk"],
        accept_multiple_files=True,
        help="Select the .vhdr, .eeg, and .vmrk files together.",
        key="brainvision_uploader",
    )

    if uploaded_files:
        # Classify uploaded files by extension
        file_map = {}
        for f in uploaded_files:
            ext = os.path.splitext(f.name)[1].lower()
            file_map[ext] = f

        # Show uploaded files
        for f in uploaded_files:
            st.info(f"📄 **{f.name}** ({f.size / 1024:.1f} KB)")

        missing = [ext for ext in (".vhdr", ".eeg", ".vmrk") if ext not in file_map]
        if missing:
            st.warning(
                f"⚠️ Missing required file(s): **{', '.join(missing)}**. "
                "Please upload all three BrainVision files (.vhdr, .eeg, .vmrk)."
            )
        else:
            basename = os.path.splitext(file_map[".vhdr"].name)[0]
            sub_dir = os.path.join(DATA_DIR, basename)
            os.makedirs(sub_dir, exist_ok=True)

            with st.spinner("⏳ Saving and loading BrainVision data..."):
                try:
                    vhdr_path = ""
                    for ext, file_obj in file_map.items():
                        save_path = os.path.join(sub_dir, file_obj.name)
                        with open(save_path, "wb") as out_f:
                            out_f.write(file_obj.getbuffer())

                        if ext == ".vhdr":
                            vhdr_path = save_path

                    logger.info(f"BrainVision files saved to: {sub_dir}")

                    raw = load_eeg_file(vhdr_path, file_type=".vhdr")
                    logger.info(f"BrainVision file loaded successfully")

                    _display_raw_info(raw, file_map[".vhdr"].name)

                    st.toast(
                        f"✅ BrainVision files saved to data/{basename}/", icon="💾"
                    )

                except Exception as e:
                    logger.error(f"Failed to load BrainVision files: {str(e)}")
                    st.error(f"❌ Failed to load BrainVision files: {e}")

# ==========================================
# --- 🕰️ Load & Manage History ---
# ==========================================
st.markdown("---")
st.markdown("### 🕰️ Load & Manage History")

if os.path.exists(DATA_DIR):
    sub_dirs = [
        d for d in os.listdir(DATA_DIR) if os.path.isdir(os.path.join(DATA_DIR, d))
    ]

    if sub_dirs:
        col_hist1, col_hist2, col_hist3 = st.columns([3, 1, 1])

        with col_hist1:
            selected_dir = st.selectbox(
                "Select a previously saved dataset:",
                options=sub_dirs,
                index=0,
                key="history_selectbox",
            )

        with col_hist2:
            st.write("")
            st.write("")
            load_clicked = st.button(
                "📥 Load", use_container_width=True, key="load_history_btn"
            )

        with col_hist3:
            st.write("")
            st.write("")
            delete_clicked = st.button(
                "🗑️ Delete", use_container_width=True, key="delete_history_btn"
            )

        # ==========================================
        # Execute logic outside of columns (full width)
        # ==========================================
        if load_clicked:
            dir_path = os.path.join(DATA_DIR, selected_dir)
            files_in_dir = os.listdir(dir_path)

            target_file = None
            target_suffix = None

            # Look for BrainVision files first
            vhdr_files = [f for f in files_in_dir if f.lower().endswith(".vhdr")]
            if vhdr_files:
                target_file = vhdr_files[0]
                target_suffix = ".vhdr"
            else:
                # Look for other supported formats
                valid_exts = [".edf", ".bdf", ".fif", ".set", ".csv"]
                for f in files_in_dir:
                    ext = os.path.splitext(f)[1].lower()
                    if ext in valid_exts:
                        target_file = f
                        target_suffix = ext
                        break

            if target_file:
                file_path = os.path.join(dir_path, target_file)
                with st.spinner(f"⏳ Loading dataset: {selected_dir}..."):
                    try:
                        logger.info(f"Loading history dataset: {selected_dir}")
                        raw = load_eeg_file(file_path, file_type=target_suffix)
                        _display_raw_info(raw, target_file)
                        st.toast(f"✅ Loaded: {selected_dir}", icon="✅")
                    except Exception as e:
                        logger.error(f"Failed to load history dataset: {str(e)}")
                        st.error(f"❌ Failed to load history dataset: {e}")
            else:
                st.error(f"❌ No valid EEG files found in folder '{selected_dir}'.")

        if delete_clicked:
            dir_path = os.path.join(DATA_DIR, selected_dir)
            try:
                shutil.rmtree(dir_path)
                logger.info(f"Deleted dataset: {selected_dir}")
                st.success(f"✅ Successfully deleted dataset: {selected_dir}")
                st.rerun()
            except Exception as e:
                logger.error(f"Failed to delete dataset: {str(e)}")
                st.error(f"❌ Deletion failed: {e}")

    else:
        st.info(
            "📂 The local 'data' folder is currently empty. Upload a file above to save it."
        )
else:
    st.info(
        "📂 The 'data' folder does not exist yet. It will be created when you upload your first file."
    )
