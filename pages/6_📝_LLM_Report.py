import streamlit as st
from utils.llm_utils import generate_eeg_report
from config import LLM_PROVIDER, OPENAI_API_KEY, GEMINI_API_KEY, OPENAI_MODEL, GEMINI_MODEL
from utils.logger import get_logger

logger = get_logger("llm_report_page")

st.set_page_config(page_title="LLM Report", page_icon="📝", layout="wide")

st.title("📝 自动分析报告生成")
st.markdown("利用大语言模型（LLM），将复杂的脑电频段特征转化为人类可读的分析报告。")

# 检查是否已经计算过 Band Power
if "band_powers" not in st.session_state or "raw" not in st.session_state:
    st.warning("⚠️ 缺少必要的分析数据。请先在 **📊 Analysis** 页面计算 Band Power。")
    st.stop()

# --- 侧边栏：API 配置和选择 ---
st.sidebar.markdown("### 🔑 LLM 设置")

# LLM 提供商选择
available_providers = []
if OPENAI_API_KEY:
    available_providers.append("🔴 OpenAI (GPT-4O Mini)")
if GEMINI_API_KEY:
    available_providers.append("🔵 Google Gemini 2.5 Flash")

if not available_providers:
    st.error(
        "❌ 未配置任何 LLM API!\n\n"
        "**设置步骤**:\n"
        "1. 创建 `.env` 文件（复制 `.env.example`）\n"
        "2. 选择一个 API 并配置:\n"
        "   - **OpenAI**: `OPENAI_API_KEY=sk-...`\n"
        "   - **Gemini**: `GEMINI_API_KEY=...`\n"
        "3. 设置 `LLM_PROVIDER=openai` 或 `LLM_PROVIDER=gemini`\n"
        "4. 重启 Streamlit 应用\n\n"
        "获取 API 密钥:\n"
        "- OpenAI: https://platform.openai.com/account/api-keys\n"
        "- Gemini: https://aistudio.google.com/app/apikey"
    )
    st.stop()

selected_provider = st.sidebar.selectbox(
    "选择 LLM 提供商",
    available_providers,
    index=0
)

# 显示当前配置
st.sidebar.markdown("### 📊 当前配置")
if "OpenAI" in selected_provider:
    st.sidebar.success(f"✅ OpenAI API 已连接")
    st.sidebar.info(f"**模型**: {OPENAI_MODEL}")
elif "Gemini" in selected_provider:
    st.sidebar.success(f"✅ Google Gemini API 已连接")
    st.sidebar.info(f"**模型**: {GEMINI_MODEL}")

st.sidebar.info(
    "**💡 提示**: 修改 `.env` 文件中的 `LLM_PROVIDER` 来设置默认提供商"
)
st.sidebar.markdown("### 🔑 API 状态")

try:
    # 尝试验证 API 连接（轻量级检查）
    from utils.llm_utils import get_llm_client

    llm = get_llm_client()
    if "OpenAI" in selected_provider:
        st.sidebar.success(f"✅ OpenAI API 已连接 ({OPENAI_MODEL})")
    else:
        st.sidebar.success(f"✅ Gemini API 已连接 ({GEMINI_MODEL})")
    api_available = True
except Exception as e:
    st.sidebar.error(f"❌ API 连接失败: {str(e)}")
    logger.error(f"API connection failed: {str(e)}")
    api_available = False

# --- 主界面 ---
st.markdown("### 📊 当前可用特征")
st.write(f"已检测到 **{len(st.session_state.raw.ch_names)}** 个通道的频段能量数据。")

col1, col2, col3 = st.columns([2, 1, 1])

with col1:
    selected_provider_display = "OpenAI (GPT-4O Mini)" if "OpenAI" in selected_provider else "Google Gemini 2.5 Flash"
    st.info(
        f"✨ 使用 {selected_provider_display} 生成专业分析报告\n\n"
        f"点击下方按钮开始生成"
    )

with col2:
    st.metric(
        "频段数",
        (
            len(st.session_state.band_powers)
            if isinstance(st.session_state.band_powers, dict)
            else "计算中"
        ),
    )

with col3:
    st.metric("通道数", len(st.session_state.raw.ch_names))

if st.button("🚀 生成智能分析报告", type="primary", disabled=not api_available):
    if not api_available:
        st.error("❌ API 未正确配置，无法生成报告")
    else:
        # 创建进度容器
        progress_placeholder = st.empty()
        status_placeholder = st.empty()
        
        try:
            # 步骤 1: 连接 API
            with progress_placeholder.container():
                st.info("📡 步骤 1/3: 正在连接 API 服务器...")
            
            logger.info(f"Generating report using {selected_provider}...")
            
            # 步骤 2: 处理数据
            with progress_placeholder.container():
                st.info("📊 步骤 2/3: 正在处理 EEG 数据...")
            
            # 步骤 3: 生成报告
            with progress_placeholder.container():
                st.info("🤖 步骤 3/3: AI 正在生成报告（预计 30-60 秒）...")
            
            with st.spinner("处理中..."):
                # 调用 LangChain 处理流
                report_content = generate_eeg_report(
                    band_powers=st.session_state.band_powers,
                    ch_names=st.session_state.raw.ch_names,
                )

                st.session_state.report_content = report_content
            
            # 清除进度提示
            progress_placeholder.empty()
            
            # 显示成功信息
            st.success(f"✅ 报告生成完毕！（使用 {selected_provider}）")
            logger.info("Report generated successfully")

        except TimeoutError as e:
            progress_placeholder.empty()
            logger.error(f"Report generation timeout: {str(e)}")
            st.error(
                f"❌ API 响应超时（60秒）\n\n"
                f"**可能原因**:\n"
                f"1. 网络连接不稳定\n"
                f"2. API 服务器响应缓慢\n"
                f"3. 数据量过大\n\n"
                f"**建议**:\n"
                f"- 检查网络连接\n"
                f"- 稍后重试\n"
                f"- 查看日志: `eeg_analysis.log`"
            )
        except ConnectionError as e:
            progress_placeholder.empty()
            logger.error(f"Network connection error: {str(e)}")
            st.error(
                f"❌ 网络连接失败\n\n"
                f"**可能原因**:\n"
                f"1. 网络断开或不稳定\n"
                f"2. API 密钥无效\n"
                f"3. 防火墙阻止连接\n\n"
                f"**建议**:\n"
                f"- 检查网络连接\n"
                f"- 验证 API 密钥\n"
                f"- 检查防火墙设置"
            )
        except Exception as e:
            progress_placeholder.empty()
            error_msg = str(e)
            logger.error(f"Report generation failed: {error_msg}")
            
            if "429" in error_msg:
                st.error(
                    f"❌ API 配额已用尽\n\n"
                    f"**原因**: 请求过于频繁\n"
                    f"**建议**: 请稍后再试"
                )
            elif "401" in error_msg or "403" in error_msg:
                st.error(
                    f"❌ API 密钥无效或权限不足\n\n"
                    f"**建议**:\n"
                    f"1. 检查 `.env` 文件中的 API 密钥\n"
                    f"2. 确认 API Key 是否过期\n"
                    f"3. 查看日志: `eeg_analysis.log`"
                )
            else:
                st.error(
                    f"❌ 生成失败！\n\n"
                    f"**错误**: {error_msg}\n\n"
                    f"**排查步骤**:\n"
                    f"1. 检查 `.env` 文件中的 API 密钥\n"
                    f"2. 检查网络连接\n"
                    f"3. 检查 API 使用额度\n"
                    f"4. 查看应用日志: `eeg_analysis.log`"
                )

# 如果报告已生成，展示并提供下载
if "report_content" in st.session_state and st.session_state.report_content is not None:
    st.markdown("---")
    # 展示报告
    st.markdown(st.session_state.report_content)

    st.markdown("---")
    # 提供 Markdown 文件下载
    if isinstance(st.session_state.report_content, str) and len(st.session_state.report_content) > 0:
        st.download_button(
            label="📥 下载报告 (Markdown)",
            data=st.session_state.report_content,
            file_name="EEG_Analysis_Report.md",
            mime="text/markdown",
        )
    else:
        st.warning("⚠️ 报告内容无效，无法下载")
