"""
LLM Utilities for Automated EEG Report Generation.
"""
import pandas as pd
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser

def generate_eeg_report(band_powers, ch_names, api_key, model_name="gpt-4o-mini"):
    """
    使用大模型生成脑电数据分析报告。
    
    Parameters:
    -----------
    band_powers : dict
        来自 st.session_state.band_powers 的频段能量数据
    ch_names : list
        通道名称列表
    api_key : str
        大模型的 API Key
    model_name : str
        使用的模型名称
        
    Returns:
    --------
    str: Markdown 格式的分析报告
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