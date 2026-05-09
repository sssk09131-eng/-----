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