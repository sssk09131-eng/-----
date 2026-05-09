import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error
import tensorflow as tf
from tensorflow.keras import layers, models, optimizers
import matplotlib.pyplot as plt
import seaborn as sns
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

# 创建序列模型
model = tf.keras.Sequential()

# 添加LSTM层（适合时间序列）
model.add(tf.keras.layers.LSTM(64, input_shape=(X_train_scaled.shape[1], 1), return_sequences=True))
model.add(tf.keras.layers.LSTM(32))
model.add(tf.keras.layers.Dense(16, activation='relu'))
model.add(tf.keras.layers.Dense(1))  # 预测单个值

# 编译模型
model.compile(optimizer='adam', loss='mse', metrics=['mae'])
    
# 训练模型
history = model.fit(
    X_train_scaled_reshaped,  # 需要调整形状为[样本数, 时间步, 特征数]
    y_train,
    epochs=50,
    batch_size=32,
    validation_split=0.2,  # 20%作为验证集
    verbose=1
)

# 绘制loss曲线
plt.figure(figsize=(12, 6))
plt.plot(history.history['loss'], label='训练loss')
plt.plot(history.history['val_loss'], label='验证loss')
plt.title('模型训练过程')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.savefig('outputs/loss_curve.png')
plt.close() 
