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