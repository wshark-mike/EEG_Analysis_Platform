"""
Preprocessing Page — Apply filters, re-referencing, and ICA.
"""

import streamlit as st
from utils.preprocessing import (
    apply_bandpass_filter,
    apply_notch_filter,
    apply_rereferencing,
    run_ica,
    apply_ica_exclusion,
    detect_bad_channels,
)
from utils.visualization import plot_raw_signals_plotly, plot_ica_components

st.set_page_config(page_title="Preprocessing", page_icon="🔧", layout="wide")

st.title("🔧 Preprocessing")

if "raw" not in st.session_state or st.session_state.raw is None:
    st.warning("⚠️ No data loaded. Please go to **Data Upload** first.")
    st.stop()

raw = st.session_state.raw

st.markdown(f"**Current data:** {len(raw.ch_names)} channels, "
            f"{raw.info['sfreq']} Hz, "
            f"{raw.n_times / raw.info['sfreq']:.1f} s")

st.markdown("---")

# --- Bad Channel Detection ---
st.markdown("### 🔍 Bad Channel Detection")

col1, col2 = st.columns(2)
with col1:
    threshold = st.slider("Z-score threshold", 1.0, 5.0, 3.0, 0.5)
with col2:
    if st.button("Detect bad channels"):
        with st.spinner("Detecting..."):
            bad_chs = detect_bad_channels(raw, threshold=threshold)
        if bad_chs:
            st.warning(f"Detected bad channels: {', '.join(bad_chs)}")
            st.session_state.bad_channels = bad_chs
        else:
            st.success("No bad channels detected.")
            st.session_state.bad_channels = []

if "bad_channels" in st.session_state and st.session_state.bad_channels:
    if st.button("Remove bad channels"):
        raw.info["bads"] = st.session_state.bad_channels
        raw.drop_channels(st.session_state.bad_channels)
        st.session_state.raw = raw
        st.success(f"Removed channels: {', '.join(st.session_state.bad_channels)}")
        st.session_state.bad_channels = []
        st.rerun()

st.markdown("---")

# --- Filtering ---
st.markdown("### 🎛️ Filtering")

filter_tab1, filter_tab2 = st.tabs(["Bandpass Filter", "Notch Filter"])

with filter_tab1:
    col1, col2 = st.columns(2)
    with col1:
        l_freq = st.number_input("Low frequency (Hz)", 0.01, 100.0, 0.1, 0.1)
    with col2:
        h_freq = st.number_input("High frequency (Hz)", 1.0, 500.0, 40.0, 1.0)

    if st.button("Apply Bandpass Filter"):
        with st.spinner("Applying bandpass filter..."):
            raw_filtered = apply_bandpass_filter(raw, l_freq=l_freq, h_freq=h_freq)
            st.session_state.raw = raw_filtered
        st.success(f"✅ Bandpass filter applied ({l_freq}–{h_freq} Hz)")
        st.rerun()

with filter_tab2:
    notch_freq = st.selectbox("Power line frequency", [50.0, 60.0])
    include_harmonics = st.checkbox("Include harmonics", value=True)

    if st.button("Apply Notch Filter"):
        freqs = [notch_freq]
        if include_harmonics:
            freqs = [notch_freq * i for i in range(1, 5)]
        with st.spinner("Applying notch filter..."):
            raw_notched = apply_notch_filter(raw, freqs=freqs)
            st.session_state.raw = raw_notched
        st.success(f"✅ Notch filter applied at {freqs} Hz")
        st.rerun()

st.markdown("---")

# --- Re-referencing ---
st.markdown("### 🔗 Re-referencing")

ref_options = ["average"] + list(raw.ch_names)
ref_type = st.selectbox("Reference type", ref_options)

if st.button("Apply Re-reference"):
    with st.spinner("Applying re-reference..."):
        raw_reref = apply_rereferencing(raw, ref_type=ref_type)
        st.session_state.raw = raw_reref
    st.success(f"✅ Re-referenced to: {ref_type}")
    st.rerun()

st.markdown("---")

# --- ICA ---
st.markdown("### 🧩 ICA Artifact Removal")

n_components = st.slider(
    "Number of ICA components",
    2, min(len(raw.ch_names), 30), min(15, len(raw.ch_names)),
)

if st.button("Run ICA"):
    with st.spinner("Running ICA (this may take a moment)..."):
        ica = run_ica(raw, n_components=n_components)
        st.session_state.ica = ica
    st.success(f"✅ ICA fitted with {n_components} components")

if "ica" in st.session_state:
    st.markdown("#### ICA Component Time Courses")
    n_display = st.slider("Components to display", 1, n_components, min(5, n_components))
    fig = plot_ica_components(st.session_state.ica, raw, n_components=n_display)
    st.pyplot(fig)

    exclude_input = st.text_input(
        "Components to exclude (comma-separated indices, e.g., 0,1,3)",
        value="",
    )

    if st.button("Apply ICA Exclusion"):
        if exclude_input.strip():
            exclude_idx = [int(x.strip()) for x in exclude_input.split(",") if x.strip()]
            with st.spinner("Applying ICA exclusion..."):
                raw_clean = apply_ica_exclusion(raw, st.session_state.ica, exclude_idx)
                st.session_state.raw = raw_clean
            st.success(f"✅ Excluded ICA components: {exclude_idx}")
            st.rerun()
        else:
            st.warning("Please enter component indices to exclude.")

st.markdown("---")

# --- Reset ---
if st.button("🔄 Reset to original data"):
    if "raw_original" in st.session_state:
        st.session_state.raw = st.session_state.raw_original.copy()
        if "ica" in st.session_state:
            del st.session_state.ica
        st.success("✅ Data reset to original.")
        st.rerun()
    else:
        st.warning("No original data stored.")

# --- Preview ---
st.markdown("---")
st.markdown("### 👁️ Current Data Preview")
fig = plot_raw_signals_plotly(
    st.session_state.raw, duration=min(5.0, raw.n_times / raw.info["sfreq"]),
    n_channels=min(5, len(raw.ch_names)),
)
st.plotly_chart(fig, use_container_width=True)