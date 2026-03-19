"""
EEG Analysis Platform - Main Application

A Streamlit-based platform for EEG data analysis, providing tools for
data loading, visualization, preprocessing, and frequency/time analysis.
"""

import streamlit as st

st.set_page_config(
    page_title="EEG Analysis Platform",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)


def main():
    st.title("🧠 EEG Analysis Platform")
    st.markdown("---")

    st.markdown(
        """
        Welcome to the **EEG Analysis Platform**! This tool helps you analyze
        electroencephalography (EEG) data through an intuitive web interface.

        ### Features

        - **📂 Data Upload** — Load EEG data from EDF, BDF, FIF, SET, or CSV files
        - **📈 Visualization** — View raw signals with interactive plots
        - **🔧 Preprocessing** — Apply filters, re-referencing, and ICA artifact removal
        - **📊 Analysis** — Compute PSD, band power, and channel connectivity

        ### Getting Started

        1. Navigate to the **Data Upload** page from the sidebar
        2. Upload your EEG data file
        3. Explore, preprocess, and analyze your data

        ### Supported File Formats

        | Format | Extension | Description |
        |--------|-----------|-------------|
        | EDF    | `.edf`    | European Data Format |
        | BDF    | `.bdf`    | BioSemi Data Format |
        | FIF    | `.fif`    | MNE-Python native format |
        | SET    | `.set`    | EEGLAB format |
        | CSV    | `.csv`    | Comma-Separated Values |
        """
    )

    # Show current data status in sidebar
    st.sidebar.markdown("### 📋 Data Status")
    if "raw" in st.session_state and st.session_state.raw is not None:
        raw = st.session_state.raw
        st.sidebar.success("✅ Data loaded")
        st.sidebar.info(
            f"**Channels:** {len(raw.ch_names)}  \n"
            f"**Sample Rate:** {raw.info['sfreq']} Hz  \n"
            f"**Duration:** {raw.n_times / raw.info['sfreq']:.1f} s"
        )

        st.sidebar.markdown("---")
        st.sidebar.markdown("### 🛤️ Processing Pipeline")
        
        if "pipeline" in st.session_state:
            for i, step in enumerate(st.session_state.pipeline):
                if i == 0:
                    st.sidebar.markdown(f"**{i}.** {step}")
                elif i == len(st.session_state.pipeline) - 1:
                    # 高亮当前最后一步
                    st.sidebar.markdown(f"👉 **{i}. {step}**")
                else:
                    st.sidebar.markdown(f"**{i}.** {step}")

        if "raw_history" in st.session_state and len(st.session_state.raw_history) > 0:
            if st.sidebar.button("↩️ 撤销上一步 (Undo)", use_container_width=True):
                # 1. 恢复上一次的数据状态
                st.session_state.raw = st.session_state.raw_history.pop()
                # 2. 删除流水线里的最后一条文字记录
                st.session_state.pipeline.pop()
                # 3. 如果 ICA 被撤销了，清理掉 ica 对象
                if "ica" in st.session_state and "ICA" in st.session_state.pipeline[-1]:
                    pass # 这里的逻辑可以根据你的具体情况精细化，目前简单重置即可
                    
                st.sidebar.success("已撤销上一步操作！")
                st.rerun()
    else:
        st.sidebar.warning("⚠️ No data loaded. Go to **Data Upload** to begin.")

    st.sidebar.markdown("---")
    st.sidebar.markdown("### 🧹 System & Memory")
    if st.sidebar.button("Clear Cache & Free Memory", use_container_width=True):
        # 1. 清理 session state 里的巨大数据对象
        for key in ["raw", "raw_original", "ica", "psd_data"]:
            if key in st.session_state:
                del st.session_state[key]
                
        # 2. 清理 @st.cache_resource 的底层缓存
        st.cache_resource.clear()
        
        st.sidebar.success("Memory cleared!")
        st.rerun() # 强制刷新页面


if __name__ == "__main__":
    main()