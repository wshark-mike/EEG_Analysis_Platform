"""
Model Inference & Training Page — Train EEGNet or run inference.
(Optimized for models/nets and models/weights structure)
"""

import os
import streamlit as st
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader

# 💡 修改点 1：从新的 nets 目录中导入 EEGNet
from models.nets.EEGNet import EEGNet

st.set_page_config(page_title="Model Inference & Training", page_icon="🤖", layout="wide")
st.title("🤖 BCI Model Training & Inference (EEGNet)")

# --- 1. 数据环境校验 ---
if "raw" not in st.session_state or st.session_state.raw is None:
    st.warning("⚠️ No data loaded. Please go to **Data Upload** first.")
    st.stop()

raw = st.session_state.raw
sfreq = int(raw.info['sfreq'])
C = len(raw.ch_names)

st.markdown(f"**Current Data:** {C} channels | **Sampling Rate:** {sfreq} Hz")
st.markdown("---")

# --- 2. 数据切分 (Epoching) 准备 ---
st.markdown("### ✂️ Data Epoching")
col_e1, col_e2 = st.columns(2)
with col_e1:
    epoch_duration = st.number_input("Trial Duration (seconds)", min_value=0.1, max_value=10.0, value=1.0, step=0.1)
with col_e2:
    st.info(f"Each trial will contain **{int(epoch_duration * sfreq)}** time samples (T).")

T = int(epoch_duration * sfreq)

# 💡 修改点 2：确保新的目录结构存在
WEIGHTS_DIR = os.path.join("models", "weights")
os.makedirs(os.path.join("models", "nets"), exist_ok=True)
os.makedirs(WEIGHTS_DIR, exist_ok=True)

@st.cache_data(show_spinner=False)
def extract_epochs(data_array, samples_per_epoch, n_classes=2):
    """提取连续脑电片段作为 Epochs"""
    channels, times = data_array.shape
    n_epochs = times // samples_per_epoch
    
    epochs_data = np.zeros((n_epochs, channels, samples_per_epoch))
    for i in range(n_epochs):
        epochs_data[i] = data_array[:, i*samples_per_epoch : (i+1)*samples_per_epoch]
        
    labels = np.random.randint(0, n_classes, size=n_epochs)
    return epochs_data, labels

with st.spinner("Extracting epochs..."):
    X_data, y_labels = extract_epochs(raw.get_data(), T)
    st.success(f"Extracted **{len(y_labels)}** trials from continuous data.")

# --- 3. 模式切换：训练 vs 推理 ---
tab_train, tab_infer = st.tabs(["🏋️ Train New Model", "🚀 Run Inference (Load Weights)"])

# ==========================================
# 🏋️ 训练模式 (Training Mode)
# ==========================================
with tab_train:
    st.markdown("#### Model Hyperparameters (EEGNet)")
    
    with st.form("train_form"):
        col_h1, col_h2, col_h3, col_h4 = st.columns(4)
        with col_h1:
            F1 = st.number_input("F1 (Temporal Filters)", min_value=4, max_value=32, value=8)
            D = st.number_input("D (Spatial Filters)", min_value=1, max_value=8, value=2)
        with col_h2:
            dropout = st.slider("Dropout Rate", 0.0, 0.9, 0.25)
            N = st.number_input("N (Number of Classes)", min_value=2, max_value=10, value=2)
        with col_h3:
            filter_len = st.number_input("Filter Length", min_value=16, max_value=128, value=64, step=16)
        with col_h4:
            epochs = st.number_input("Training Epochs", min_value=1, max_value=500, value=20, step=5)
            lr = st.number_input("Learning Rate", min_value=0.0001, max_value=0.1, value=0.001, format="%.4f")
            
        train_btn = st.form_submit_button("🔥 Start Training", use_container_width=True)
        
    if train_btn:
        if T < 32:
            st.error("Error: Trial duration is too short. 'T' must be at least 32 samples for EEGNet pooling layers.")
        else:
            X_tensor = torch.FloatTensor(X_data)
            y_tensor = torch.LongTensor(y_labels)
            dataset = TensorDataset(X_tensor, y_tensor)
            dataloader = DataLoader(dataset, batch_size=32, shuffle=True)
            
            model = EEGNet(F1=F1, D=D, C=C, T=T, dropout=dropout, N=N, filter_len=filter_len)
            criterion = nn.CrossEntropyLoss()
            optimizer = optim.Adam(model.parameters(), lr=lr)
            
            st.markdown("#### Training Progress")
            progress_bar = st.progress(0)
            status_text = st.empty()
            loss_chart = st.empty()
            
            model.train()
            loss_history = []
            
            for epoch in range(epochs):
                running_loss = 0.0
                for inputs, labels in dataloader:
                    optimizer.zero_grad()
                    outputs = model(inputs)
                    loss = criterion(outputs, labels)
                    loss.backward()
                    optimizer.step()
                    running_loss += loss.item()
                
                avg_loss = running_loss / len(dataloader)
                loss_history.append(avg_loss)
                
                progress_bar.progress((epoch + 1) / epochs)
                status_text.text(f"Epoch [{epoch+1}/{epochs}] - Loss: {avg_loss:.4f}")
                
                loss_df = pd.DataFrame({"Loss": loss_history})
                loss_chart.line_chart(loss_df)
                
            # 💡 修改点 3：将权重保存到 models/weights 目录下
            save_name = f"eegnet_C{C}_T{T}_N{N}.pth"
            save_path = os.path.join(WEIGHTS_DIR, save_name)
            torch.save(model.state_dict(), save_path)
            st.success(f"🎉 Training Complete! Model weights saved to: `{save_path}`")

# ==========================================
# 🚀 推理模式 (Inference Mode)
# ==========================================
with tab_infer:
    st.markdown("#### Run Pre-trained Model")
    
    # 💡 修改点 4：只扫描 models/weights 目录下的 .pth 文件
    model_files = [f for f in os.listdir(WEIGHTS_DIR) if f.endswith(".pth")] if os.path.exists(WEIGHTS_DIR) else []
    
    if not model_files:
        st.info("No saved model weights found in `models/weights/`. Please train a model first.")
    else:
        col_i1, col_i2 = st.columns([3, 1])
        with col_i1:
            selected_model = st.selectbox("Select Model Weights:", model_files)
        with col_i2:
            st.write("")
            st.write("")
            infer_btn = st.button("🚀 Run Inference", use_container_width=True)
            
        if infer_btn:
            # 💡 修改点 5：拼接正确的读取路径
            model_path = os.path.join(WEIGHTS_DIR, selected_model)
            
            with st.spinner("Loading model and running inference..."):
                try:
                    N_infer = int(selected_model.split('_N')[1].split('.pth')[0]) if '_N' in selected_model else 2
                    
                    # 实例化模型并加载权重
                    model_infer = EEGNet(F1=8, D=2, C=C, T=T, dropout=0.25, N=N_infer, filter_len=64)
                    model_infer.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
                    model_infer.eval()
                    
                    # 运行前向传播
                    X_tensor = torch.FloatTensor(X_data)
                    with torch.no_grad():
                        outputs = model_infer(X_tensor)
                        probabilities = torch.softmax(outputs, dim=1).numpy()
                        predictions = np.argmax(probabilities, axis=1)
                        
                    st.success("✅ Inference completed!")
                    
                    results_df = pd.DataFrame({
                        "Trial ID": [f"Trial {i+1}" for i in range(len(predictions))],
                        "Predicted Class": [f"Class {p}" for p in predictions],
                        "Confidence": [f"{probabilities[i][predictions[i]] * 100:.2f}%" for i in range(len(predictions))]
                    })
                    
                    st.dataframe(results_df.head(50), use_container_width=True)
                    st.caption(f"Showing first 50 results out of {len(predictions)} trials.")
                    
                except Exception as e:
                    st.error(f"Failed to run inference. Error: {e}\n\nMake sure the current data shape (C={C}, T={T}) matches the network structure.")