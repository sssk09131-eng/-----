# main.py
import re
import os
from m1_data_processing import load_processed_data
from m2_visualization import generate_hourly_demand, generate_area_heatmap
from m3_prediction import predict_demand, predict_fare

def main():
    print("欢迎使用智能交通出行数据分析系统")
    print("您可以询问时段查询、区域排名、需求预测、费用估算等问题")
    print("输入'exit'退出系统")
    
    # 加载处理好的数据（M1的结果）
    df = load_processed_data()
    
    while True:
        user_input = input("\n请输入您的问题: ")
        if user_input.lower() == 'exit':
            print("感谢使用，再见！")
            break
        
        # 处理用户问题
        handle_user_query(user_input, df)

def handle_user_query(query, df):
    """处理用户查询的主要函数"""
    # 1. 分析用户意图
    intent, params = analyze_intent(query)
    
    # 2. 根据意图调用相应模块
    if intent == "time_query":
        process_time_query(params, df)
    elif intent == "area_ranking":
        process_area_ranking(params, df)
    # ... 其他意图处理
    else:
        print("无法理解您的问题，请尝试更明确的提问。")
        print("支持的问题类型包括：时段查询、区域排名、需求预测、费用估算等。")

if __name__ == "__main__":
    main()