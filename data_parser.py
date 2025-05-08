from datetime import datetime
import pandas as pd


def parse_input_to_series(input_str):
    """ 解析输入字符串为日期 -> 净值的 Pandas Series """
    parsed = {}
    for item in input_str.split(","):
        date_str, value_str = item.split("^")
        parsed[date_str] = float(value_str)
    dates = sorted(parsed.keys())
    values = [parsed[d] for d in dates]
    date_index = pd.to_datetime(dates)
    return pd.Series(values, index=date_index)