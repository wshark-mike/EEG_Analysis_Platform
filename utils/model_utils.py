"""
Deep Learning Utilities for EEG Inference.
"""
import torch
import torch.nn as nn
import numpy as np

# 这是一个高度简化的 EEGNet 骨架示例
# 实际使用时，请替换为你自己线下训练时使用的完整网络结构
class SimpleEEGNet(nn.Module):
    def __init__(self, n_channels, n_classes):
        super(SimpleEEGNet, self).__init__()
        self.conv1 = nn.Conv2d(1, 16, (1, 64), padding='same')
        self.batchnorm1 = nn.BatchNorm2d(16)
        self.depthwise = nn.Conv2d(16, 32, (n_channels, 1), groups=16)
        self.flatten = nn.Flatten()
        # 这里的 Linear 维度需要根据实际的时间步长(samples)计算，这里暂用占位符
        self.fc = nn.LazyLinear(n_classes) 

    def forward(self, x):
        x = self.conv1(x)
        x = self.batchnorm1(x)
        x = self.depthwise(x)
        x = nn.functional.elu(x)
        x = self.flatten(x)
        x = self.fc(x)
        return x

@torch.no_grad() # 推理阶段不需要计算梯度，节省内存
def run_bci_inference(data_array, model_path, n_channels, n_classes=2):
    """
    运行模型推理
    
    Parameters:
    -----------
    data_array : np.ndarray
        预处理后的脑电片段数据，形状应为 (n_trials, n_channels, n_samples)
    model_path : str
        预训练权重的本地路径
    n_channels : int
        通道数
        
    Returns:
    --------
    predictions : np.ndarray
        类别预测结果
    probabilities : np.ndarray
        各个类别的概率
    """
    # 1. 初始化模型并加载权重
    model = SimpleEEGNet(n_channels=n_channels, n_classes=n_classes)
    
    try:
        # 尝试加载权重 (如果你有真实的 .pth 文件)
        model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
    except Exception as e:
        # 为了演示，如果找不到权重，我们就用随机初始化的模型跑一遍
        print(f"Warning: Could not load weights from {model_path}. Using untrained model. Error: {e}")
    
    model.eval()

    # 2. 数据格式转换 (MNE Numpy Array -> PyTorch Tensor)
    # EEGNet 通常需要 4D 输入: (batch_size, 1, channels, time_steps)
    tensor_data = torch.FloatTensor(data_array).unsqueeze(1) 
    
    # 3. 前向传播
    outputs = model(tensor_data)
    
    # 4. 计算概率和预测类别
    probabilities = torch.softmax(outputs, dim=1).numpy()
    predictions = np.argmax(probabilities, axis=1)
    
    return predictions, probabilities