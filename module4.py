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

def analyze_intent(query):
    
    # 转换为小写便于匹配
    query = query.lower()
    
    # 1. 时段查询
    if re.search(r'(时段|时间段|高峰|小时|几点|什么时候).*?需求|订单|出行', query):
        return "time_query", extract_time_params(query)
    
    # 2. 区域排名
    elif re.search(r'(区域|地方|地点|位置).*?(排名|最多|最高|热门)', query):
        return "area_ranking", extract_area_params(query)
    
    # 3. 需求预测
    elif re.search(r'(预测|预计|估计|将会).*?(需求|订单|出行|量)', query):
        return "demand_prediction", extract_prediction_params(query)
    
    # 4. 费用估算
    elif re.search(r'(费用|车费|价格|多少钱|花费).*?(估算|预测|预计)', query):
        return "fare_estimate", extract_fare_params(query)
    
    # 5. 其他问题类型（至少再加一种）
    elif re.search(r'(拥堵|交通状况|效率|速度)', query):
        return "traffic_condition", extract_traffic_params(query)
    
    # 无法匹配的情况
    else:
        return None, None

def extract_time_params(query):
    
    params = {}
    
    # 提取小时信息
    hour_match = re.search(r'(\d{1,2})\s*点|(\d{1,2})\s*时', query)
    if hour_match:
        params['hour'] = int(hour_match.group(1) or hour_match.group(2))
    
    # 提取工作日/周末信息
    if '工作日' in query:
        params['day_type'] = 'weekday'
    elif '周末' in query:
        params['day_type'] = 'weekend'
    
    return params
def process_time_query(params, df):
    
    # 调用M2中的可视化函数
    if 'hour' in params and 'day_type' in params:
        chart_path = generate_hourly_demand(df, params['hour'], params['day_type'])
        print(f"您查询的{params['day_type']} {params['hour']}点的出行需求如下:")
        print(f"图表已生成: {chart_path}")
    else:
        # 生成默认分析
        chart_path = generate_hourly_demand(df)
        print("为您展示全天出行需求时间规律:")
        print(f"图表已生成: {chart_path}")

def process_area_ranking(params, df):
    
    # 调用M2中的区域热度分析
    chart_path = generate_area_heatmap(df, top_n=10)
    print("上客量最高的TOP 10区域及高峰时段分布:")
    print(f"图表已生成: {chart_path}")
    
    # 添加额外信息
    print("\n区域排名详情:")
    top_areas = df['pickup_area'].value_counts().head(5)
    for i, (area, count) in enumerate(top_areas.items(), 1):
        print(f"{i}. {area}: {count}单")

def process_demand_prediction(params, df):
    
    # 从参数中提取区域和时间
    area = params.get('area', '中心区域')
    time = params.get('time', '明天10点')
    
    # 调用M3中的预测函数
    predicted_demand = predict_demand(df, area, time)
    
    print(f"预测{area}在{time}的出行需求量为: {predicted_demand:.1f}单")
    print("预测基于历史数据和机器学习模型")

def process_fare_estimate(params, df):
    """处理费用估算查询"""
    # 从参数中提取距离、时段等
    distance = params.get('distance', 5.0)
    time = params.get('time', '工作日早高峰')
    
    # 调用M3中的费用预测
    estimated_fare = predict_fare(df, distance, time)
    
    print(f"行程距离{distance}公里，在{time}的预计费用为: ¥{estimated_fare:.2f}")
    print("费用估算基于历史数据和机器学习模型")

def handle_unmatched_query(query):
    
    print("\n无法理解您的问题，可能是以下原因:")
    print("1. 问题表述不够清晰")
    print("2. 问题类型不在支持范围内")
    
    print("\n支持的问题类型包括:")
    print("- 时段查询: '工作日几点出行需求最高？', '周末高峰时段是几点？'")
    print("- 区域排名: '上客量最高的区域是哪些？', '热门区域排名'")
    print("- 需求预测: '明天10点中心区域的出行需求是多少？'")
    print("- 费用估算: '5公里行程在工作日早高峰需要多少钱？'")
    print("- 交通状况: '哪些区域拥堵最严重？', '高峰时段的交通效率如何？'")
    
    print("\n请尝试更明确的提问，或参考上述示例。")