import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
def load_and_explore_data(file_path):
    
     # 尝试多种编码方式加载数据
    encodings = ['utf-8', 'gbk', 'latin1']
    for encoding in encodings:
        try:
            data= pd.read_csv(file_path, encoding=encoding)
            print(f"成功使用{encoding}编码加载数据")
            break
        except Exception as e:
            print(f"使用{encoding}编码加载失败: {str(e)}")
            if encoding == encodings[-1]:
                raise
    # 显示数据的前几行
    print("数据预览:")
    print(data.head())
    
    # 显示数据的基本信息
    print("\n数据基本信息:")
    print(data.info())
    
    # 显示数据的统计描述
    print("\n数据统计描述:")
    print(data.describe())
    
    return data

#生成数据质量报告
def generate_data_quality_report(data):
    print("\n数据质量报告:")
    # 检查缺失值
    missing_values = data.isnull().sum()
    print("\n缺失值统计:")
    print(missing_values)
    #缺失率    
    missing_rate = (missing_values / len(data)) * 100
    print(f"\n缺失率: {missing_rate.round(2).tolist()}")
    
    #异常值检测
    numeric_cols = data.select_dtypes(include=[np.number]).columns
    outliers = {}
    
    for col in numeric_cols:
        # IQR方法检测异常值
        Q1 = data[col].quantile(0.25)
        Q3 = data[col].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        
        # 统计异常值数量
        outlier_count = ((data[col] < lower_bound) | (data[col] > upper_bound)).sum()
        outliers[col] = {
            'lower_bound': lower_bound,
            'upper_bound': upper_bound,
            'outlier_count': outlier_count,
            'outlier_percent': (outlier_count / len(data)) * 100
        }
    
    # 检查数据类型
    print("\n数据类型统计:")
    print(data.dtypes)

#生成报告
    quality_report = {
    'missing_values': data.isnull().sum(),
    'missing_rate': (data.isnull().sum() / len(data)) * 100,
    'outliers': outliers,
    'data_types': data.dtypes
}
    return quality_report



#数据清洗策略
def clean_data(df, quality_report):
    """
    清洗数据，处理缺失值和异常值
    
    参数:
    df: 原始DataFrame
    quality_report: 数据质量报告
    
    返回:
    cleaned_df: 清洗后的DataFrame
    """
    # 创建副本避免修改原始数据
    cleaned_df = df.copy()
    
    # 1. 处理缺失值
    # 策略: 对于缺失率>30%的列直接删除（保留核心特征）
    
    for col, missing_pct in quality_report['missing_percent'].items():
        if missing_pct > 30:
            # 删除缺失率过高的列
            cleaned_df.drop(col, axis=1, inplace=True)
            print(f"删除缺失率{missing_pct:.2f}%的特征: {col} (缺失率>30%)")
        elif missing_pct > 0:
            # 根据数据类型选择填充策略
            if cleaned_df[col].dtype in [np.float64, np.int64]:
                # 数值型特征：用中位数填充（对异常值不敏感）
                median_val = cleaned_df[col].median()
                cleaned_df[col].fillna(median_val, inplace=True)
                print(f"数值特征 {col} 用中位数 {median_val} 填充缺失值")
            else:
                # 分类型特征：用众数填充
                mode_val = cleaned_df[col].mode()[0]
                cleaned_df[col].fillna(mode_val, inplace=True)
                print(f"分类型特征 {col} 用众数 {mode_val} 填充缺失值")
    
    # 2. 处理异常值
    # 策略: 对于行程时间等关键特征，采用边界截断法处理异常值
    
    for col, info in quality_report['outliers'].items():
        if 'trip_duration' in col.lower() or 'time' in col.lower():
            # 对行程时间特征进行边界截断
            cleaned_df[col] = np.where(cleaned_df[col] < info['lower_bound'], 
                                      info['lower_bound'], 
                                      cleaned_df[col])
            cleaned_df[col] = np.where(cleaned_df[col] > info['upper_bound'], 
                                      info['upper_bound'], 
                                      cleaned_df[col])
            print(f"对{col}进行边界截断处理，范围[{info['lower_bound']:.2f}, {info['upper_bound']:.2f}]")
    
    # 3. 处理重复值
    duplicates = cleaned_df.duplicated().sum()
    if duplicates > 0:
        cleaned_df.drop_duplicates(inplace=True)
        print(f"删除{duplicates}条重复数据")
    
    return cleaned_df



#特征工程

def feature_engineering(df):
    """
    特征工程：从行程时间中提取特征，并创建衍生特征
    
    参数:
    df: 清洗后的DataFrame
    
    返回:
    engineered_df: 特征工程后的DataFrame
    """
    engineered_df = df.copy()
    
    # 确保日期时间列存在且格式正确
    if 'trip_start_time' in engineered_df.columns:
        # 转换为datetime类型
        engineered_df['trip_start_time'] = pd.to_datetime(engineered_df['trip_start_time'])
        
        # 1. 提取小时特征
        engineered_df['hour'] = engineered_df['trip_start_time'].dt.hour
        print("提取行程开始时间的小时特征")
        
        # 2. 提取星期特征
        engineered_df['day_of_week'] = engineered_df['trip_start_time'].dt.dayofweek
        print("提取行程开始时间的星期特征 (0=周一, 6=周日)")
        
        # 3. 判断是否高峰时段（假设早高峰7-9点，晚高峰17-19点）
        engineered_df['is_peak_hour'] = engineered_df['hour'].apply(
            lambda x: 1 if (7 <= x <= 9) or (17 <= x <= 19) else 0
        )
        print("创建是否高峰时段特征 (1=高峰, 0=非高峰)")
    
    # 4. 衍生特征1：行程速度（假设距离已知）
    if 'trip_distance' in engineered_df.columns and 'trip_duration' in engineered_df.columns:
        # 避免除以零的情况
        engineered_df['trip_speed'] = engineered_df['trip_distance'] / (engineered_df['trip_duration'] + 1e-6)
        print("创建行程速度特征 (距离/时间)")
    
    # 5. 衍生特征2：是否工作日
    if 'day_of_week' in engineered_df.columns:
        engineered_df['is_workday'] = engineered_df['day_of_week'].apply(
            lambda x: 1 if x < 5 else 0
        )
        print("创建是否工作日特征 (1=工作日, 0=周末)")
    
    # 6. 衍生特征3：行程时间段（早、中、晚、夜）
    if 'hour' in engineered_df.columns:
        engineered_df['time_of_day'] = engineered_df['hour'].apply(
            lambda x: 'morning' if 5 <= x < 12 else
                      'afternoon' if 12 <= x < 18 else
                      'evening' if 18 <= x < 22 else 'night'
        )
        print("创建行程时间段特征 (morning/afternoon/evening/night)")
    
    # 7. 衍生特征4：行程是否在节假日（需要节假日数据）
    # 这里简化处理，假设没有节假日数据，可以基于星期特征近似
    if 'day_of_week' in engineered_df.columns:
        engineered_df['is_holiday_approx'] = engineered_df['day_of_week'].apply(
            lambda x: 1 if x >= 5 else 0
        )
        print("创建节假日近似特征 (基于周末，1=周末/节假日, 0=工作日)")
    
    return engineered_df

def main_data_processing(file_path):
    """
    主数据处理流程
    
    参数:
    file_path: 数据文件路径
    """
    print("="*50)
    print("开始数据处理流程")
    print("="*50)
    
    # 步骤1: 加载数据
    print("\n步骤1: 加载数据")
    df = load_and_explore_data(file_path)
    
    # 步骤2: 生成数据质量报告
    print("\n步骤2: 生成数据质量报告")
    quality_report = generate_data_quality_report(df)
    
    # 步骤3: 数据清洗
    print("\n步骤3: 数据清洗")
    cleaned_df = clean_data(df, quality_report)
    
    # 步骤4: 特征工程
    print("\n步骤4: 特征工程")
    final_df = feature_engineering(cleaned_df)
    
    # 保存处理后的数据
    final_df.to_csv('processed_data.csv', index=False)
    print("\n数据处理完成!")
    print(f"处理后数据形状: {final_df.shape}")
    print("处理后的数据已保存至 'processed_data.csv'")
    
    return final_df

# 使用示例
if __name__ == "__main__":
    # 替换为你的数据文件路径
    file_path = "traffic_data.csv"
    processed_data = main_data_processing(file_path)
