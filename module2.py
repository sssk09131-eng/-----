# 示例代码框架 - 你需要在M1完成后执行
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

# 确保输出目录存在
os.makedirs('outputs', exist_ok=True)

# 加载M1处理后的数据
df = pd.read_csv('processed_data.csv')

# 1. 按小时分析
hourly_demand = df.groupby('hour')['order_id'].count().reset_index(name='order_count')
# 可视化
plt.figure(figsize=(12, 6))
sns.lineplot(x='hour', y='order_count', data=hourly_demand)
plt.title('每小时平均订单量')
plt.xlabel('小时')
plt.ylabel('订单量')
# 保存图表
plt.savefig('outputs/hourly_demand.png')
plt.close()

# 2. 按工作日/周末分析
# 首先确保有工作日/周末特征（M1中应该已创建）
weekday_weekend_demand = df.groupby(['day_of_week', 'is_workday'])['order_id'].count().reset_index(name='order_count')
# 可视化
plt.figure(figsize=(10, 6))
sns.barplot(x='day_of_week', y='order_count', hue='is_workday', data=weekday_weekend_demand)
plt.title('工作日 vs 周末订单量对比')
# 保存图表
plt.savefig('outputs/weekday_weekend_demand.png')
plt.close()