"""
LLM Utilities for Automated EEG Report Generation.
Supports both OpenAI and Google Gemini APIs.
"""

import pandas as pd
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from utils.logger import get_logger
from config import (
    LLM_PROVIDER, 
    OPENAI_API_KEY, 
    OPENAI_MODEL,
    GEMINI_API_KEY,
    GEMINI_MODEL,
    MAX_API_TOKENS_PER_SESSION
)

logger = get_logger("llm_utils")


def get_llm_client():
    """
    Get configured LLM client based on LLM_PROVIDER setting.
    
    Returns
    -------
    llm : ChatOpenAI or ChatGoogleGenerativeAI
        Configured language model client
        
    Raises
    ------
    ValueError: If provider not configured or API key missing
    """
    if LLM_PROVIDER == "openai":
        if not OPENAI_API_KEY:
            raise ValueError(
                "❌ OPENAI_API_KEY not configured!\n"
                "Set in .env: OPENAI_API_KEY=sk-..."
            )
        
        from langchain_openai import ChatOpenAI
        logger.info(f"Using OpenAI: {OPENAI_MODEL}")
        return ChatOpenAI(
            model=OPENAI_MODEL,
            openai_api_key=OPENAI_API_KEY,
            temperature=0.3,
            request_timeout=60,
            max_retries=2
        )
    
    elif LLM_PROVIDER == "gemini":
        if not GEMINI_API_KEY:
            raise ValueError(
                "❌ GEMINI_API_KEY not configured!\n"
                "Set in .env: GEMINI_API_KEY=..."
            )
        
        from langchain_google_genai import ChatGoogleGenerativeAI
        logger.info(f"Using Gemini: {GEMINI_MODEL}")
        return ChatGoogleGenerativeAI(
            model=GEMINI_MODEL,
            google_api_key=GEMINI_API_KEY,
            temperature=0.3,
            request_timeout=60,
            max_retries=2
        )
    
    else:
        raise ValueError(
            f"❌ Unknown LLM_PROVIDER: {LLM_PROVIDER}\n"
            f"Must be 'openai' or 'gemini'"
        )


def estimate_tokens(band_powers, ch_names):
    """
    Rough estimation of API tokens that will be used.
    1 token ≈ 4 characters in English
    """
    data_size = len(ch_names) * len(band_powers) * 10  # chars
    prompt_size = 800  # system prompt
    response_size = 1500  # estimated response
    total_chars = data_size + prompt_size + response_size
    estimated_tokens = total_chars // 4
    return estimated_tokens


def generate_eeg_report(band_powers, ch_names):
    """
    Generate EEG analysis report using configured LLM (OpenAI or Gemini).
    
    Parameters:
    -----------
    band_powers : dict
        Band power data from st.session_state.band_powers
    ch_names : list
        List of channel names

    Returns:
    --------
    str: Analysis report in Markdown format
        
    Raises:
    -------
    ValueError: If API key is not configured or data is invalid
    """
    if not band_powers or not ch_names:
        raise ValueError("Invalid band_powers or ch_names data")
    
    # Estimate token usage
    estimated_tokens = estimate_tokens(band_powers, ch_names)
    if estimated_tokens > MAX_API_TOKENS_PER_SESSION:
        logger.warning(
            f"Estimated token usage {estimated_tokens} exceeds limit "
            f"{MAX_API_TOKENS_PER_SESSION}"
        )
        raise ValueError(
            f"Estimated API usage too high: {estimated_tokens} tokens\n"
            f"Limit: {MAX_API_TOKENS_PER_SESSION} tokens"
        )
    
    logger.info(
        f"Generating report with {len(ch_names)} channels, "
        f"estimated {estimated_tokens} tokens (Provider: {LLM_PROVIDER})"
    )
    
    # Format data for LLM
    df = pd.DataFrame(band_powers, index=ch_names)
    mean_powers = df.mean().to_dict()
    
    data_summary = f"Number of channels: {len(ch_names)}\n"
    data_summary += "Global average band power:\n"
    for band, power in mean_powers.items():
        data_summary += f"- {band}: {power:.4f}\n"
    
    # Build prompt template (same for both providers)
    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                """You are an expert cognitive neuroscientist and EEG analyst.
Analyze the provided EEG band power data and generate a professional report.

Include:
1. Data Overview
2. Band-specific Analysis (interpret what each band indicates)
3. Key Findings & Recommendations

Use Markdown formatting. Keep language scientifically rigorous but accessible.
IMPORTANT: Clearly state this is a feature analysis, not clinical diagnosis.""",
            ),
            ("human", "Analyze this EEG data:\n{data_summary}\nGenerate the report."),
        ]
    )

    try:
        logger.info("🔄 正在连接 API 服务器...")
        llm = get_llm_client()
        
        logger.info("🤖 正在处理数据...")
        chain = prompt | llm | StrOutputParser()
        
        logger.info("📝 正在生成分析报告，这可能需要 30-60 秒...")
        report = chain.invoke({"data_summary": data_summary})

        logger.info(f"✅ 报告生成完成 ({LLM_PROVIDER})")
        return report

    except TimeoutError as e:
        logger.error(f"API 超时 ({LLM_PROVIDER}): {str(e)}")
        raise ValueError(
            f"❌ API 响应超时（60秒）\n"
            f"可能原因:\n"
            f"1. 网络连接不稳定\n"
            f"2. API 服务器响应缓慢\n"
            f"3. 数据量过大\n\n"
            f"建议: 检查网络连接，稍后重试"
        ) from e
    except ConnectionError as e:
        logger.error(f"网络连接错误 ({LLM_PROVIDER}): {str(e)}")
        raise ValueError(
            f"❌ 网络连接失败\n"
            f"可能原因:\n"
            f"1. 网络断开或不稳定\n"
            f"2. API 密钥无效\n"
            f"3. 防火墙阻止连接\n\n"
            f"建议: 检查网络和 API 密钥配置"
        ) from e
    except Exception as e:
        error_msg = str(e)
        logger.error(f"报告生成失败 ({LLM_PROVIDER}): {error_msg}")
        
        if "429" in error_msg:
            raise ValueError(
                f"❌ API 请求过于频繁\n"
                f"API 配额已用尽，请稍后再试"
            ) from e
        elif "401" in error_msg or "403" in error_msg:
            raise ValueError(
                f"❌ API 密钥无效或权限不足\n"
                f"请检查 .env 文件中的 API 密钥"
            ) from e
        else:
            raise ValueError(
                f"❌ 报告生成失败: {error_msg}\n\n"
                f"排查步骤:\n"
                f"1. 检查 .env 文件中的 API 密钥\n"
                f"2. 检查网络连接\n"
                f"3. 检查 API 额度使用情况\n"
                f"4. 查看日志: eeg_analysis.log"
            ) from e
