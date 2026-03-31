"""
Visualization Page — View raw EEG signals and basic plots.
(Optimized for fast page switching and lazy rendering)
"""

import streamlit as st
import numpy as np
import plotly.graph_objects as go
from utils.visualization import plot_raw_signals_plotly, plot_raw_signals

st.set_page_config(page_title="Visualization", page_icon="📈", layout="wide")

st.title("📈 Signal Visualization")

# --- 1. 数据校验 ---
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

# --- 2. 核心控制区 (使用 st.form 阻断自动刷新) ---
st.markdown("### 🎛️ Plot Controls")

max_duration = raw.n_times / raw.info["sfreq"]

# 使用 st.form 包裹所有的滑动条和输入框
with st.form("viz_controls"):
    col1, col2, col3, col4 = st.columns(4)

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

    # 只有点击这个按钮，才会触发下方的绘图计算
    submitted = st.form_submit_button("🎨 生成 / 更新波形图 (Generate Plot)", use_container_width=True)


# --- 3. 绘图计算与缓存逻辑 ---
st.markdown("### 🌊 EEG Signal Plot")

# 如果用户点击了“生成”按钮，进行繁重的绘图计算，并把结果存下来
if submitted:
    with st.spinner("Generating high-density plot... Please wait."):
        if plot_type == "Interactive (Plotly)":
            fig = plot_raw_signals_plotly(
                raw, duration=duration, n_channels=n_channels, start=start_time,
            )
            # 缓存生成的图表和类型
            st.session_state.viz_fig_main = fig
            st.session_state.viz_type = "plotly"
        else:
            fig = plot_raw_signals(
                raw, duration=duration, n_channels=n_channels, start=start_time,
            )
            # 缓存生成的图表和类型
            st.session_state.viz_fig_main = fig
            st.session_state.viz_type = "matplotlib"

# --- 4. 前端渲染逻辑 ---
# 无论页面怎么切换，只要缓存里有图，直接拿出来秒级渲染
if "viz_fig_main" in st.session_state:
    if st.session_state.viz_type == "plotly":
        st.plotly_chart(st.session_state.viz_fig_main, use_container_width=True)
    else:
        st.pyplot(st.session_state.viz_fig_main)
else:
    st.info("👆 请调整上方的参数，并点击 **生成 / 更新波形图** 来查看脑电信号。")


# --- 5. 单通道独立视图 (同样进行按需渲染优化) ---
st.markdown("---")
st.markdown("### 🔍 Individual Channel View")

# 使用列布局分离选择器和生成按钮
col_ch1, col_ch2 = st.columns([3, 1])
with col_ch1:
    selected_channel = st.selectbox("Select a specific channel to inspect", raw.ch_names)
with col_ch2:
    st.write("") # 占位对齐
    st.write("")
    render_single_ch = st.button("👁️ 查看单通道 (View Channel)", use_container_width=True)

# 只有点击按钮才去切片数据并画图
if render_single_ch and selected_channel:
    with st.spinner(f"Extracting data for {selected_channel}..."):
        ch_idx = raw.ch_names.index(selected_channel)
        sfreq = raw.info["sfreq"]
        start_sample = int(start_time * sfreq)
        end_sample = int((start_time + duration) * sfreq)
        end_sample = min(end_sample, raw.n_times)

        data = raw.get_data(picks=[ch_idx], start=start_sample, stop=end_sample)
        times = np.arange(start_sample, end_sample) / sfreq

        # 为了防止单通道高频数据也卡顿，加入一个简单的动态降采样保护
        max_points = 5000
        step = max(1, len(times) // max_points)
        
        data_plot = data[0][::step]
        times_plot = times[::step]

        fig_single = go.Figure()
        fig_single.add_trace(
            go.Scatter(x=times_plot, y=data_plot * 1e6, mode="lines", name=selected_channel)
        )
        
        title_suffix = f" (Downsampled to {len(times_plot)} pts)" if step > 1 else ""
        fig_single.update_layout(
            title=f"Channel: {selected_channel}{title_suffix}",
            xaxis_title="Time (s)",
            yaxis_title="Amplitude (µV)",
            height=400,
        )
        
        # 将单通道图表也存入缓存
        st.session_state.viz_fig_single = fig_single

# 渲染单通道缓存图表
if "viz_fig_single" in st.session_state:
    st.plotly_chart(st.session_state.viz_fig_single, use_container_width=True)