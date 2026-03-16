"""
Analysis Page — Frequency analysis, band power, and connectivity.
"""

import streamlit as st
import numpy as np
from utils.analysis import (
    compute_psd,
    compute_band_power,
    compute_connectivity,
    get_freq_bands,
)
from utils.visualization import plot_psd, plot_band_power, plot_connectivity_matrix

st.set_page_config(page_title="Analysis", page_icon="📊", layout="wide")

st.title("📊 Analysis")

if "raw" not in st.session_state or st.session_state.raw is None:
    st.warning("⚠️ No data loaded. Please go to **Data Upload** first.")
    st.stop()

raw = st.session_state.raw
st.markdown(f"**Current data:** {len(raw.ch_names)} channels, "
            f"{raw.info['sfreq']} Hz, "
            f"{raw.n_times / raw.info['sfreq']:.1f} s")

st.markdown("---")

# --- PSD Analysis ---
st.markdown("### 📉 Power Spectral Density (PSD)")

col1, col2, col3 = st.columns(3)
with col1:
    fmin = st.number_input("Min frequency (Hz)", 0.1, 100.0, 0.5, 0.5)
with col2:
    fmax = st.number_input("Max frequency (Hz)", 1.0, 500.0, 50.0, 1.0)
with col3:
    psd_method = st.selectbox("PSD method", ["welch", "multitaper"])

if st.button("Compute PSD"):
    with st.spinner("Computing PSD..."):
        psd_data, freqs = compute_psd(raw, fmin=fmin, fmax=fmax, method=psd_method)
        st.session_state.psd_data = psd_data
        st.session_state.psd_freqs = freqs
    st.success("✅ PSD computed")

if "psd_data" in st.session_state:
    log_scale = st.checkbox("Log scale (dB)", value=True)
    fig = plot_psd(
        st.session_state.psd_data,
        st.session_state.psd_freqs,
        raw.ch_names,
        log_scale=log_scale,
    )
    st.plotly_chart(fig, use_container_width=True)

st.markdown("---")

# --- Band Power ---
st.markdown("### 📊 Band Power Analysis")

bands = get_freq_bands()
st.markdown("**Standard EEG frequency bands:**")
band_info_cols = st.columns(len(bands))
for i, (band, (lo, hi)) in enumerate(bands.items()):
    band_info_cols[i].metric(band, f"{lo}–{hi} Hz")

if st.button("Compute Band Power"):
    with st.spinner("Computing band power..."):
        band_powers = compute_band_power(raw, bands=bands)
        st.session_state.band_powers = band_powers
    st.success("✅ Band power computed")

if "band_powers" in st.session_state:
    fig = plot_band_power(st.session_state.band_powers, raw.ch_names)
    st.plotly_chart(fig, use_container_width=True)

    # Show numerical values
    with st.expander("Show numerical values"):
        import pandas as pd
        df = pd.DataFrame(st.session_state.band_powers, index=raw.ch_names)
        st.dataframe(df, use_container_width=True)

st.markdown("---")

# --- Connectivity ---
st.markdown("### 🔗 Channel Connectivity")

conn_method = st.selectbox("Connectivity method", ["correlation"])

if st.button("Compute Connectivity"):
    with st.spinner("Computing connectivity matrix..."):
        conn_matrix, ch_names = compute_connectivity(raw, method=conn_method)
        st.session_state.conn_matrix = conn_matrix
        st.session_state.conn_ch_names = ch_names
    st.success("✅ Connectivity computed")

if "conn_matrix" in st.session_state:
    fig = plot_connectivity_matrix(
        st.session_state.conn_matrix, st.session_state.conn_ch_names,
    )
    st.plotly_chart(fig, use_container_width=True)

st.markdown("---")

# --- Data Export ---
st.markdown("### 💾 Export Results")

export_format = st.selectbox("Export format", ["CSV", "FIF"])

if st.button("Export processed data"):
    import tempfile
    import os

    with st.spinner("Preparing export..."):
        if export_format == "CSV":
            import pandas as pd
            data = raw.get_data().T
            df = pd.DataFrame(data, columns=raw.ch_names)
            df.insert(0, "time", np.arange(len(df)) / raw.info["sfreq"])
            csv_data = df.to_csv(index=False)
            st.download_button(
                label="📥 Download CSV",
                data=csv_data,
                file_name="eeg_processed.csv",
                mime="text/csv",
            )
        elif export_format == "FIF":
            with tempfile.NamedTemporaryFile(
                delete=False, suffix="_raw.fif"
            ) as tmp:
                raw.save(tmp.name, overwrite=True, verbose=False)
                with open(tmp.name, "rb") as f:
                    fif_data = f.read()
                os.unlink(tmp.name)
            st.download_button(
                label="📥 Download FIF",
                data=fif_data,
                file_name="eeg_processed_raw.fif",
                mime="application/octet-stream",
            )