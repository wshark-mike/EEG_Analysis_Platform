"""
Visualization Page — View raw EEG signals and basic plots.
"""

import streamlit as st
from utils.visualization import plot_raw_signals_plotly, plot_raw_signals

st.set_page_config(page_title="Visualization", page_icon="📈", layout="wide")

st.title("📈 Signal Visualization")

if "raw" not in st.session_state or st.session_state.raw is None:
    st.warning("⚠️ No data loaded. Please go to **Data Upload** first.")
    st.stop()

raw = st.session_state.raw

st.markdown(f"**File:** {st.session_state.get('filename', 'Unknown')}")
st.markdown(
    f"**Channels:** {len(raw.ch_names)} | "
    f"**Sample Rate:** {raw.info['sfreq']} Hz | "
    f"**Duration:** {raw.n_times / raw.info['sfreq']:.1f} s"
)

st.markdown("---")

# Controls
col1, col2, col3, col4 = st.columns(4)

max_duration = raw.n_times / raw.info["sfreq"]

with col1:
    start_time = st.number_input(
        "Start time (s)", min_value=0.0, max_value=max(0.0, max_duration - 1.0),
        value=0.0, step=1.0,
    )

with col2:
    duration = st.number_input(
        "Duration (s)", min_value=1.0, max_value=min(30.0, max_duration),
        value=min(10.0, max_duration), step=1.0,
    )

with col3:
    n_channels = st.number_input(
        "Number of channels", min_value=1, max_value=len(raw.ch_names),
        value=min(8, len(raw.ch_names)), step=1,
    )

with col4:
    plot_type = st.selectbox("Plot type", ["Interactive (Plotly)", "Static (Matplotlib)"])

# Generate plot
st.markdown("### EEG Signal Plot")

with st.spinner("Generating plot..."):
    if plot_type == "Interactive (Plotly)":
        fig = plot_raw_signals_plotly(
            raw, duration=duration, n_channels=n_channels, start=start_time,
        )
        st.plotly_chart(fig, use_container_width=True)
    else:
        fig = plot_raw_signals(
            raw, duration=duration, n_channels=n_channels, start=start_time,
        )
        st.pyplot(fig)

# Channel selector
st.markdown("---")
st.markdown("### Individual Channel View")
selected_channel = st.selectbox("Select channel", raw.ch_names)

if selected_channel:
    import plotly.graph_objects as go
    import numpy as np

    ch_idx = raw.ch_names.index(selected_channel)
    sfreq = raw.info["sfreq"]
    start_sample = int(start_time * sfreq)
    end_sample = int((start_time + duration) * sfreq)
    end_sample = min(end_sample, raw.n_times)

    data = raw.get_data(picks=[ch_idx], start=start_sample, stop=end_sample)
    times = np.arange(start_sample, end_sample) / sfreq

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(x=times, y=data[0] * 1e6, mode="lines", name=selected_channel)
    )
    fig.update_layout(
        title=f"Channel: {selected_channel}",
        xaxis_title="Time (s)",
        yaxis_title="Amplitude (µV)",
        height=400,
    )
    st.plotly_chart(fig, use_container_width=True)