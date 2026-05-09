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

# 评估神经网络模型
y_pred = model.predict(X_test_scaled_reshaped)
mae_nn = mean_absolute_error(y_test, y_pred)
rmse_nn = np.sqrt(mean_squared_error(y_test, y_pred))

# 保存评估报告
with open('outputs/nn_report.txt', 'w') as f:
    f.write(f"神经网络模型评估报告\n")
    f.write(f"MAE: {mae_nn:.4f}\n")
    f.write(f"RMSE: {rmse_nn:.4f}\n")
# 训练随机森林模型
rf = RandomForestRegressor(n_estimators=100, random_state=42)
rf.fit(X_train_scaled, y_train)

# 评估随机森林
y_pred_rf = rf.predict(X_test_scaled)
mae_rf = mean_absolute_error(y_test, y_pred_rf)
rmse_rf = np.sqrt(mean_squared_error(y_test, y_pred_rf))

# 对比结果
print(f"神经网络 - MAE: {mae_nn:.4f}, RMSE: {rmse_nn:.4f}")
print(f"随机森林 - MAE: {mae_rf:.4f}, RMSE: {rmse_rf:.4f}")

# 可视化对比
plt.figure(figsize=(12, 6))
plt.plot(y_test.values, label='实际值')
plt.plot(y_pred, label='神经网络预测')
plt.plot(y_pred_rf, label='随机森林预测')
plt.title('预测结果对比')
plt.xlabel('时间')
plt.ylabel('需求量')
plt.legend()
plt.savefig('outputs/prediction_comparison.png')
plt.close()