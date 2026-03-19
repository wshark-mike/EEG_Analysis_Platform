"""
Preprocessing Page — Apply filters, re-referencing, ICA, and bad channel rejection.
(Optimized with Pipeline Tracking, Undo Support, and Lazy Preview)
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

# ==========================================
# 💡 核心辅助函数：记录操作流水线并备份数据
# ==========================================
def backup_and_log(action_name):
    # 1. 备份数据 (限制最多存 3 步，防内存溢出)
    if "raw_history" not in st.session_state:
        st.session_state.raw_history = []
    if len(st.session_state.raw_history) >= 3:
        st.session_state.raw_history.pop(0)
    # 必须用深拷贝
    st.session_state.raw_history.append(st.session_state.raw.copy())
    
    # 2. 记录文字流水线
    if "pipeline" not in st.session_state:
        st.session_state.pipeline = [f"📂 原始数据 ({st.session_state.get('filename', 'Unknown')})"]
    st.session_state.pipeline.append(action_name)

# ==========================================
# --- 🔍 Bad Channel Detection ---
# ==========================================
st.markdown("### 🔍 Bad Channel Detection")
col1, col2 = st.columns([2, 1])
with col1:
    threshold = st.slider("Z-score threshold", 1.0, 5.0, 3.0, 0.5)
with col2:
    st.write("")
    st.write("")
    if st.button("Detect bad channels", use_container_width=True):
        with st.spinner("Detecting..."):
            bad_chs = detect_bad_channels(raw, threshold=threshold)
        if bad_chs:
            st.session_state.bad_channels = bad_chs
            st.rerun()
        else:
            st.success("No bad channels detected.")
            st.session_state.bad_channels = []

if "bad_channels" in st.session_state and st.session_state.bad_channels:
    st.warning(f"Detected bad channels: {', '.join(st.session_state.bad_channels)}")
    if st.button("🗑️ Remove bad channels", type="primary"):
        with st.spinner("Removing..."):
            backup_and_log(f"🚫 剔除坏道 ({len(st.session_state.bad_channels)}个)")
            
            raw.info["bads"] = st.session_state.bad_channels
            raw.drop_channels(st.session_state.bad_channels)
            st.session_state.raw = raw
            
            st.session_state.bad_channels = []
        st.success("Bad channels removed.")
        st.rerun()

st.markdown("---")

# ==========================================
# --- 🎛️ Filtering ---
# ==========================================
st.markdown("### 🎛️ Filtering")
filter_tab1, filter_tab2 = st.tabs(["Bandpass Filter", "Notch Filter"])

with filter_tab1:
    col1, col2, col3 = st.columns([2, 2, 1])
    with col1:
        l_freq = st.number_input("Low frequency (Hz)", 0.01, 100.0, 0.1, 0.1)
    with col2:
        h_freq = st.number_input("High frequency (Hz)", 1.0, 500.0, 40.0, 1.0)
    with col3:
        st.write("")
        st.write("")
        if st.button("Apply Bandpass", use_container_width=True):
            with st.spinner("Applying bandpass filter..."):
                backup_and_log(f"🎛️ 带通滤波 ({l_freq}-{h_freq} Hz)")
                raw_filtered = apply_bandpass_filter(st.session_state.raw, l_freq=l_freq, h_freq=h_freq)
                st.session_state.raw = raw_filtered
            st.success(f"✅ Bandpass filter applied")
            st.rerun()

with filter_tab2:
    col1, col2, col3 = st.columns([2, 2, 1])
    with col1:
        notch_freq = st.selectbox("Power line frequency", [50.0, 60.0])
    with col2:
        include_harmonics = st.checkbox("Include harmonics", value=True)
    with col3:
        st.write("")
        if st.button("Apply Notch", use_container_width=True):
            freqs = [notch_freq * i for i in range(1, 5)] if include_harmonics else [notch_freq]
            with st.spinner("Applying notch filter..."):
                backup_and_log(f"🎛️ 陷波滤波 ({notch_freq} Hz)")
                raw_notched = apply_notch_filter(st.session_state.raw, freqs=freqs)
                st.session_state.raw = raw_notched
            st.success(f"✅ Notch filter applied")
            st.rerun()

st.markdown("---")

# ==========================================
# --- 🔗 Re-referencing ---
# ==========================================
st.markdown("### 🔗 Re-referencing")
col1, col2 = st.columns([3, 1])
with col1:
    ref_options = ["average"] + list(raw.ch_names)
    ref_type = st.selectbox("Reference type", ref_options)
with col2:
    st.write("")
    st.write("")
    if st.button("Apply Re-reference", use_container_width=True):
        with st.spinner("Applying re-reference..."):
            backup_and_log(f"🔗 重参考 ({ref_type})")
            raw_reref = apply_rereferencing(st.session_state.raw, ref_type=ref_type)
            st.session_state.raw = raw_reref
        st.success(f"✅ Re-referenced to: {ref_type}")
        st.rerun()

st.markdown("---")

# ==========================================
# --- 🧩 ICA Artifact Removal ---
# ==========================================
st.markdown("### 🧩 ICA Artifact Removal")
col1, col2 = st.columns([3, 1])
with col1:
    n_components = st.slider("Number of ICA components", 2, min(len(raw.ch_names), 30), min(15, len(raw.ch_names)))
with col2:
    st.write("")
    st.write("")
    if st.button("Run ICA", use_container_width=True):
        with st.spinner("Running ICA (this may take a moment)..."):
            ica = run_ica(st.session_state.raw, n_components=n_components)
            st.session_state.ica = ica
        st.success(f"✅ ICA fitted with {n_components} components")

if "ica" in st.session_state:
    st.markdown("#### ICA Components")
    n_display = st.slider("Components to display", 1, n_components, min(5, n_components))
    fig = plot_ica_components(st.session_state.ica, st.session_state.raw, n_components=n_display)
    st.pyplot(fig)

    col_ex1, col_ex2 = st.columns([3, 1])
    with col_ex1:
        exclude_input = st.text_input("Components to exclude (comma-separated, e.g., 0,1)", value="")
    with col_ex2:
        st.write("")
        st.write("")
        if st.button("Apply ICA Exclusion", use_container_width=True):
            if exclude_input.strip():
                exclude_idx = [int(x.strip()) for x in exclude_input.split(",") if x.strip()]
                with st.spinner("Applying ICA exclusion..."):
                    backup_and_log(f"🧩 去除 ICA 成分 ({exclude_idx})")
                    raw_clean = apply_ica_exclusion(st.session_state.raw, st.session_state.ica, exclude_idx)
                    st.session_state.raw = raw_clean
                st.success("✅ Excluded ICA components")
                st.rerun()
            else:
                st.warning("Please enter component indices.")

st.markdown("---")

# ==========================================
# --- 👁️ Lazy Current Data Preview (防止卡顿) ---
# ==========================================
col_rst, col_prev = st.columns(2)
with col_rst:
    if st.button("🔄 Reset to original data", use_container_width=True):
        if "raw_original" in st.session_state:
            st.session_state.raw = st.session_state.raw_original.copy()
            st.session_state.pipeline = [f"📂 原始数据 ({st.session_state.get('filename', 'Unknown')})"]
            st.session_state.raw_history = []
            if "ica" in st.session_state:
                del st.session_state.ica
            st.success("✅ Data reset to original.")
            st.rerun()

st.markdown("### 👁️ Current Data Preview")
# 默认折叠，阻断页面切换时的自动重绘
with st.expander("点击展开数据预览 (Click to generate preview plot)", expanded=False):
    if st.button("🎨 生成/刷新预览图 (Generate Preview)"):
        with st.spinner("Rendering preview..."):
            fig = plot_raw_signals_plotly(
                st.session_state.raw, duration=min(5.0, raw.n_times / raw.info["sfreq"]),
                n_channels=min(5, len(raw.ch_names)),
            )
            st.session_state.preview_fig = fig
            
    if "preview_fig" in st.session_state:
        st.plotly_chart(st.session_state.preview_fig, use_container_width=True)