"""LLM utilities for automated EEG report generation.

This module provides a LangChain-based pipeline that sends EEG frequency-band
power statistics to a large language model (e.g., GPT-4o-mini) and receives a
structured, Markdown-formatted clinical summary.

Notes
-----
Requires ``langchain-core``, ``langchain-openai``, and a valid OpenAI API key.
The generated report is for informational purposes only and must **not** be
used as a clinical diagnostic tool.
"""

from typing import Dict, List

import pandas as pd
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser


def generate_eeg_report(
    band_powers: Dict[str, List[float]],
    ch_names: List[str],
    api_key: str,
    model_name: str = "gpt-4o-mini",
) -> str:
    """Generate an EEG analysis report using a large language model.

    Summarises the provided frequency-band power data into a structured,
    Markdown-formatted report written by a virtual neuroscience expert.

    Parameters
    ----------
    band_powers : dict
        Mapping from frequency-band name (e.g. ``"Alpha"``) to a list/array
        of per-channel power values.  Typically obtained from
        ``utils.analysis.compute_band_power``.
    ch_names : list of str
        Channel names corresponding to the entries in each ``band_powers``
        value array.
    api_key : str
        OpenAI-compatible API key used to authenticate the LLM request.
    model_name : str, optional
        Name of the chat model to use.  Defaults to ``"gpt-4o-mini"``.

    Returns
    -------
    report : str
        Markdown-formatted analysis report produced by the language model.

    Raises
    ------
    Exception
        Any network or API error raised by the underlying LangChain client.
    """
    # 1. 数据预处理：将字典转换为易于大模型阅读的文本格式
    df = pd.DataFrame(band_powers, index=ch_names)
    # 计算每个频段的全局平均能量，找出最显著的特征
    mean_powers = df.mean().to_dict()
    
    # 将统计数据格式化为字符串
    data_summary = f"通道总数: {len(ch_names)}\n"
    data_summary += "全脑各频段平均能量:\n"
    for band, power in mean_powers.items():
        data_summary += f"- {band}: {power:.4f}\n"
    
    # 2. 构建 LangChain 提示词模板
    prompt = ChatPromptTemplate.from_messages([
        ("system", """你是一位资深的认知神经科学专家和脑电（EEG）分析师。
你的任务是根据用户提供的脑电频段能量（Band Power）统计数据，撰写一份结构化、专业的初步分析报告。
报告需要包含以下部分：
1. 数据总览
2. 频段特征分析（解读各频段能量的相对高低代表了什么生理或心理状态）
3. 初步结论与建议
请使用 Markdown 格式排版，语言通俗易懂但保持科学严谨。注意：明确标明这仅为数据特征报告，不能作为临床诊断依据。"""),
        ("human", "这是本次提取的脑电数据特征：\n{data_summary}\n请为我生成分析报告。")
    ])

    # 3. 初始化模型实例
    llm = ChatOpenAI(temperature=0.3, model=model_name, openai_api_key=api_key)

    # 4. 构建 LCEL 处理链并执行
    chain = prompt | llm | StrOutputParser()
    
    # 触发调用
    report = chain.invoke({"data_summary": data_summary})
    return report