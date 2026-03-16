"""
Data Upload Page — Load EEG data files into the platform.
"""

import os
import tempfile
import streamlit as st
from utils.data_loader import load_eeg_file, get_raw_info, get_supported_formats

st.set_page_config(page_title="Data Upload", page_icon="📂", layout="wide")

st.title("📂 Data Upload")
st.markdown("Upload your EEG data file to begin analysis.")

# Show supported formats
formats = get_supported_formats()
with st.expander("Supported file formats"):
    for ext, desc in formats.items():
        st.markdown(f"- **{ext}** — {desc}")

# File uploader
uploaded_file = st.file_uploader(
    "Choose an EEG data file",
    type=["edf", "bdf", "fif", "set", "csv"],
    help="Upload EEG data in one of the supported formats.",
)

if uploaded_file is not None:
    st.info(f"📄 File: **{uploaded_file.name}** ({uploaded_file.size / 1024:.1f} KB)")

    # Save to temp file for MNE to read
    suffix = os.path.splitext(uploaded_file.name)[1].lower()

    with st.spinner("Loading EEG data..."):
        try:
            with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
                tmp.write(uploaded_file.getbuffer())
                tmp_path = tmp.name

            raw = load_eeg_file(tmp_path, file_type=suffix)

            # Store in session state
            st.session_state.raw = raw
            st.session_state.raw_original = raw.copy()
            st.session_state.filename = uploaded_file.name

            st.success("✅ Data loaded successfully!")

            # Display info
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

        except Exception as e:
            st.error(f"❌ Failed to load file: {e}")
        finally:
            # Clean up temp file
            if "tmp_path" in dir() and os.path.exists(tmp_path):
                os.unlink(tmp_path)

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