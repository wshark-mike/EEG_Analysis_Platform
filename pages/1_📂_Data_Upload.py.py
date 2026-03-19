"""
Data Upload Page — Load EEG data files into the platform.
"""

import os
import shutil
import streamlit as st
from utils.data_loader import load_eeg_file, get_raw_info, get_supported_formats

DATA_DIR = "data"
os.makedirs(DATA_DIR, exist_ok=True)

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

    st.session_state.pipeline = [f"📂 原始数据 ({filename})"]
    st.session_state.raw_history = []

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

        basename = os.path.splitext(uploaded_file.name)[0]
        suffix = os.path.splitext(uploaded_file.name)[1].lower()
        
        sub_dir = os.path.join(DATA_DIR, basename)
        os.makedirs(sub_dir, exist_ok=True)

        with st.spinner("Loading EEG data..."):
            try:
                save_path = os.path.join(sub_dir, uploaded_file.name)
                with open(save_path, "wb") as f:
                    f.write(uploaded_file.getbuffer())

                raw = load_eeg_file(save_path, file_type=suffix)
                _display_raw_info(raw, uploaded_file.name)

                st.toast(f"Saved to data/{basename}/", icon="💾")

            except Exception as e:
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
            
            with st.spinner("Saving and loading BrainVision data..."):
                try:
                    vhdr_path = ""
                    for ext, file_obj in file_map.items():
                        save_path = os.path.join(sub_dir, file_obj.name)
                        with open(save_path, "wb") as out_f:
                            out_f.write(file_obj.getbuffer())

                        if ext == ".vhdr":
                            vhdr_path = save_path

                    raw = load_eeg_file(vhdr_path, file_type=".vhdr")
                    _display_raw_info(raw, file_map[".vhdr"].name)
                    
                    st.toast(f"BrainVision files saved to data/{basename}/", icon="💾")

                except Exception as e:
                    st.error(f"❌ Failed to load BrainVision files: {e}")

# ==========================================
# --- 🕰️ Load & Manage History ---
# ==========================================
st.markdown("---")
st.markdown("### 🕰️ Load & Manage History")

if os.path.exists(DATA_DIR):
    sub_dirs = [d for d in os.listdir(DATA_DIR) if os.path.isdir(os.path.join(DATA_DIR, d))]
    
    if sub_dirs:
        col_hist1, col_hist2, col_hist3 = st.columns([3, 1, 1])
        
        with col_hist1:
            selected_dir = st.selectbox(
                "Select a previously saved dataset:",
                options=sub_dirs,
                index=0
            )
            
        with col_hist2:
            st.write("") 
            st.write("")
            # 只在这里捕获按钮的点击状态，不写执行逻辑
            load_clicked = st.button("📥 加载 (Load)", use_container_width=True)
                        
        with col_hist3:
            st.write("") 
            st.write("")
            # 同样只捕获状态
            delete_clicked = st.button("🗑️ 删除 (Delete)", use_container_width=True)
            
        # ==========================================
        # 💡 [关键修改]：把执行逻辑移到分栏外面，占据全宽
        # ==========================================
        if load_clicked:
            dir_path = os.path.join(DATA_DIR, selected_dir)
            files_in_dir = os.listdir(dir_path)
            
            target_file = None
            target_suffix = None
            
            vhdr_files = [f for f in files_in_dir if f.lower().endswith('.vhdr')]
            if vhdr_files:
                target_file = vhdr_files[0]
                target_suffix = ".vhdr"
            else:
                valid_exts = [".edf", ".bdf", ".fif", ".set", ".csv"]
                for f in files_in_dir:
                    ext = os.path.splitext(f)[1].lower()
                    if ext in valid_exts:
                        target_file = f
                        target_suffix = ext
                        break
            
            if target_file:
                file_path = os.path.join(dir_path, target_file)
                with st.spinner(f"Loading dataset {selected_dir}..."):
                    try:
                        raw = load_eeg_file(file_path, file_type=target_suffix)
                        # 现在这个函数会在页面的主区域（全宽）渲染，不会再被挤压了！
                        _display_raw_info(raw, target_file)
                        st.toast(f"Loaded dataset: {selected_dir}", icon="✅")
                    except Exception as e:
                        st.error(f"❌ Failed to load history dataset: {e}")
            else:
                st.error(f"❌ No valid EEG files found in folder '{selected_dir}'.")

        if delete_clicked:
            dir_path = os.path.join(DATA_DIR, selected_dir)
            try:
                import shutil
                shutil.rmtree(dir_path)
                st.success(f"✅ 成功删除数据集 {selected_dir} 及其所有文件。")
                st.rerun()
            except Exception as e:
                st.error(f"❌ Deletion failed: {e}")

    else:
        st.info("📂 The local 'data' folder is currently empty. Upload a file above to save it.")
else:
    st.info("📂 The 'data' folder does not exist yet. It will be created when you upload your first file.")