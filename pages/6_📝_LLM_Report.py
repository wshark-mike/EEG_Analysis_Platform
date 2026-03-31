import streamlit as st
from utils.llm_utils import generate_eeg_report

st.set_page_config(page_title="LLM Report", page_icon="📝", layout="wide")

st.title("📝 自动分析报告生成")
st.markdown("利用大语言模型（LLM），将复杂的脑电频段特征转化为人类可读的分析报告。")

# 检查是否已经计算过 Band Power
if "band_powers" not in st.session_state or "raw" not in st.session_state:
    st.warning("⚠️ 缺少必要的分析数据。请先在 **📊 Analysis** 页面计算 Band Power。")
    st.stop()

# --- 侧边栏：API 密钥配置 ---
st.sidebar.markdown("### 🔑 API 设置")
api_key = st.sidebar.text_input(
    "OpenAI API Key", 
    type="password", 
    help="你的 Key 只会用于本次会话，不会被保存在服务器上。"
)

model_choice = st.sidebar.selectbox(
    "选择大模型", 
    ["gpt-4o-mini", "gpt-4o", "gpt-3.5-turbo"]
)

# --- 主界面 ---
st.markdown("### 📊 当前可用特征")
st.write(f"已检测到 **{len(st.session_state.raw.ch_names)}** 个通道的频段能量数据。")

if st.button("🚀 生成智能分析报告", type="primary"):
    if not api_key:
        st.error("❌ 请先在左侧边栏输入有效的 API Key！")
    else:
        with st.spinner("🧠 专家大模型正在解读数据，请稍候..."):
            try:
                # 调用 LangChain 处理流
                report_content = generate_eeg_report(
                    band_powers=st.session_state.band_powers,
                    ch_names=st.session_state.raw.ch_names,
                    api_key=api_key,
                    model_name=model_choice
                )
                
                st.session_state.report_content = report_content
                st.success("✅ 报告生成完毕！")
                
            except Exception as e:
                st.error(f"生成失败，请检查 API Key 或网络连接。详细错误：\n{e}")

# 如果报告已生成，展示并提供下载
if "report_content" in st.session_state:
    st.markdown("---")
    # 展示报告
    st.markdown(st.session_state.report_content)
    
    st.markdown("---")
    # 提供 Markdown 文件下载
    st.download_button(
        label="📥 下载报告 (Markdown)",
        data=st.session_state.report_content,
        file_name="EEG_Analysis_Report.md",
        mime="text/markdown"
    )