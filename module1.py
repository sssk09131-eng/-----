import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
def load_and_explore_data(file_path):
    # 加载数据
    data = pd.read_csv(file_path)
    
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