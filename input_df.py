import pandas as pd

def create_df(input_str):
    # 分割字符串并准备数据列表
    data_entries = input_str.split(',')
    parsed_data = []

    for entry in data_entries:
        date_part, value_part = entry.split('^')
        parsed_data.append({
            'prod_reg_code': 'P001',  # 固定产品代码
            'date': pd.to_datetime(date_part, format='%Y%m%d'),
            'net_val': int(value_part)
        })
        # 添加 P002 的数据
        parsed_data.append({
            'prod_reg_code': 'P002',  # 新的产品代码
            'date': pd.to_datetime(date_part, format='%Y%m%d'),
            'net_val': int(value_part)  # 净值保持不变
        })

    # 创建DataFrame
    df = pd.DataFrame(parsed_data)

    return df