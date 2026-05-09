import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error
import torch
import torch.nn as nn
import torch.optim as optim

df = pd.read_csv('processed_data.csv')
# 按时间排序（确保数据按时间顺序）
df = df.sort_values('timestamp')

# 选择特征和目标
features = ['hour', 'day_of_week', 'is_peak_hour', 'area_1_demand', 'area_2_demand', ...]
target = 'next_hour_demand'

# 按时间划分（8:2）
split_idx = int(0.8 * len(df))
X_train, X_test = df[features][:split_idx], df[features][split_idx:]
y_train, y_test = df[target][:split_idx], df[target][split_idx:]

# 标准化特征（对神经网络很重要）
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

class DemandPredictor(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers):
        super(DemandPredictor, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, 1)
        
    def forward(self, x):
        # LSTM处理
        out, _ = self.lstm(x)
        # 取最后一个时间步的输出
        out = self.fc(out[:, -1, :])
        return out