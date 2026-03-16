"""
Data Upload Page — Load EEG data files into the platform.
"""

import os
import tempfile
import streamlit as st
from utils.data_loader import (
    load_eeg_file,
    load_brainvision_files,
    get_raw_info,
    get_supported_formats,
)

st.set_page_config(page_title="Data Upload", page_icon="📂", layout="wide")

st.title("📂 Data Upload")
st.markdown("Upload your EEG data file to begin analysis.")

# Show supported formats
formats = get_supported_formats()
with st.expander("Supported file formats"):
    for ext, desc in formats.items():
        st.markdown(f"- **{ext}** — {desc}")


def _display_raw_info(raw, filename):
    """Store loaded data in session state and display info metrics."""
    st.session_state.raw = raw
    st.session_state.raw_original = raw.copy()
    st.session_state.filename = filename

    st.success("✅ Data loaded successfully!")

    info = get_raw_info(raw)
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Channels", info["n_channels"])
    col2.metric("Sample Rate", f"{info['sfreq']} Hz")
    col3.metric("Duration", f"{info['duration_sec']:.1f} s")
    col4.metric("Samples", f"{info['n_samples']:,}")

    st.markdown("#### Channel Information")
    ch_data = {
        "Channel Name": info["ch_names"],
        "Type": info["ch_types"],
    }
    st.dataframe(ch_data, use_container_width=True)


# --- Upload mode selector ---
upload_mode = st.radio(
    "Select upload mode",
    ["Single file (EDF / BDF / FIF / SET / CSV)", "BrainVision (.vhdr + .eeg + .vmrk)"],
    horizontal=True,
)

if upload_mode == "Single file (EDF / BDF / FIF / SET / CSV)":
    # --- Single-file uploader ---
    uploaded_file = st.file_uploader(
        "Choose an EEG data file",
        type=["edf", "bdf", "fif", "set", "csv"],
        help="Upload EEG data in one of the supported formats.",
    )

    if uploaded_file is not None:
        st.info(f"📄 File: **{uploaded_file.name}** ({uploaded_file.size / 1024:.1f} KB)")

        suffix = os.path.splitext(uploaded_file.name)[1].lower()

        with st.spinner("Loading EEG data..."):
            try:
                with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
                    tmp.write(uploaded_file.getbuffer())
                    tmp_path = tmp.name

                raw = load_eeg_file(tmp_path, file_type=suffix)
                _display_raw_info(raw, uploaded_file.name)

            except Exception as e:
                st.error(f"❌ Failed to load file: {e}")
            finally:
                if "tmp_path" in dir() and os.path.exists(tmp_path):
                    os.unlink(tmp_path)

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
            with st.spinner("Loading BrainVision data..."):
                try:
                    raw = load_brainvision_files(
                        vhdr_buffer=file_map[".vhdr"].getbuffer(),
                        eeg_buffer=file_map[".eeg"].getbuffer(),
                        vmrk_buffer=file_map[".vmrk"].getbuffer(),
                    )
                    _display_raw_info(raw, file_map[".vhdr"].name)

                except Exception as e:
                    st.error(f"❌ Failed to load BrainVision files: {e}")

# Option to use sample data
st.markdown("---")
st.markdown("### Or use sample data")
if st.button("Load MNE sample data (auditory/visual)"):
    with st.spinner("Downloading and loading sample data... This may take a moment."):
        try:
            import mne
            sample_data_path = mne.datasets.sample.data_path()
            raw_fname = os.path.join(
                sample_data_path, "MEG", "sample", "sample_audvis_raw.fif"
            )
            raw = load_eeg_file(raw_fname, file_type=".fif")

            # Pick only EEG channels for simplicity
            raw.pick_types(eeg=True)

            st.session_state.raw = raw
            st.session_state.raw_original = raw.copy()
            st.session_state.filename = "sample_audvis_raw.fif"

            st.success("✅ Sample data loaded!")

            info = get_raw_info(raw)
            col1, col2, col3, col4 = st.columns(4)
            col1.metric("Channels", info["n_channels"])
            col2.metric("Sample Rate", f"{info['sfreq']} Hz")
            col3.metric("Duration", f"{info['duration_sec']:.1f} s")
            col4.metric("Samples", f"{info['n_samples']:,}")

        except Exception as e:
            st.error(f"❌ Failed to load sample data: {e}")