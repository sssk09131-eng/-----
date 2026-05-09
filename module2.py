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

# 1. 找出TOP 10区域
# 假设数据中有 pickup_area 和 dropoff_area 列
top_areas = df['pickup_area'].value_counts().head(10).reset_index()
top_areas.columns = ['area', 'pickup_count']

# 2. 分析高峰时段分布
# 为每个区域分析高峰时段
area_peak_hours = df.groupby(['pickup_area', 'hour'])['order_id'].count().reset_index(name='order_count')
# 只关注TOP 10区域
area_peak_hours = area_peak_hours[area_peak_hours['pickup_area'].isin(top_areas['area'])]

# 可视化
plt.figure(figsize=(14, 8))
sns.lineplot(x='hour', y='order_count', hue='pickup_area', data=area_peak_hours)
plt.title('TOP 10区域高峰时段分布')
# 保存图表
plt.savefig('outputs/area_peak_hours.png')
plt.close()

# 3. 区域热度热力图（可选）
# 创建区域和时段的交叉表
heatmap_data = area_peak_hours.pivot('pickup_area', 'hour', 'order_count')
plt.figure(figsize=(12, 8))
sns.heatmap(heatmap_data, annot=True, fmt=".0f", cmap="YlGnBu")
plt.title('区域热度热力图')
# 保存图表
plt.savefig('outputs/area_heatmap.png')
plt.close()



# 1. 距离-车费关系
plt.figure(figsize=(10, 6))
sns.scatterplot(x='trip_distance', y='fare', data=df)
plt.title('行程距离与车费关系')
# 保存图表
plt.savefig('outputs/distance_fare.png')
plt.close()

# 2. 时段-车费关系
plt.figure(figsize=(10, 6))
sns.boxplot(x='hour', y='fare', data=df)
plt.title('不同时段车费分布')
# 保存图表
plt.savefig('outputs/hour_fare.png')
plt.close()

# 3. 乘客人数-车费关系（如果有相关数据）
if 'passenger_count' in df.columns:
    plt.figure(figsize=(10, 6))
    sns.boxplot(x='passenger_count', y='fare', data=df)
    plt.title('乘客人数与车费关系')
    # 保存图表
    plt.savefig('outputs/passenger_fare.png')
    plt.close()

    # 1. 选择特征进行聚类
# 例如：小时、是否高峰、行程距离、是否工作日等
clustering_features = ['hour', 'is_peak_hour', 'trip_distance', 'is_workday']
X = df[clustering_features]

# 2. 进行聚类（简化版，实际需要标准化和选择K值）
from sklearn.cluster import KMeans
kmeans = KMeans(n_clusters=4, random_state=42)
df['cluster'] = kmeans.fit_predict(X)

# 3. 可视化聚类结果
plt.figure(figsize=(12, 8))
sns.scatterplot(x='trip_distance', y='fare', hue='cluster', data=df, palette='viridis')
plt.title('基于距离和车费的出行模式聚类')
# 保存图表
plt.savefig('outputs/cluster_analysis.png')
plt.close()

# 4. 分析每个聚类的特征
cluster_profile = df.groupby('cluster').agg({
    'hour': 'mean',
    'trip_distance': 'mean',
    'fare': 'mean',
    'is_peak_hour': 'mean'
}).reset_index()
print("聚类特征分析:")
print(cluster_profile)