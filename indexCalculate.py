# _*_ coding: utf-8 _*_
"""
@Time : 2025/5/7 15:40
@Auth : Derek
@File : indexCalculate.py
@IDE  : PyCharm
"""
import calendar
from datetime import timedelta

from pyspark.sql import SparkSession
from pyspark.sql.functions import *
from pyspark.sql.types import *
import datetime
import bisect
import logging
import os
import numpy as np
import pandas as pd
import argparse

logger = logging.getLogger()

INPUT_STR = "20250301^6,20250302^6,20250303^2,20250304^2,20250305^8,20250306^5,20250307^8,20250308^3,20250309^1,20250310^3,20250311^3,20250312^9,20250313^7,20250314^4,20250315^1,20250316^1,20250317^7,20250318^3,20250319^7,20250320^3,20250321^8,20250322^7,20250323^1,20250324^3,20250325^2,20250326^8,20250327^9,20250328^1,20250329^1,20250330^3,20250331^7,20250401^6"
WINDOWS = [7, 14, 30, 90, 182, 365, 730, 1095, 0] # 窗口周期


class ReturnRiskIndexCalculator:
    def __init__(self, input_str):
        self.input_str = input_str
        self.parsed_dict = self._parse_input()

    def _parse_input(self):
        """ 解析输入字符串为日期 -> 净值的字典 """
        parsed = {}
        for item in self.input_str.split(","):
            date_str, value_str = item.split("^")
            parsed[date_str] = float(value_str)
        return parsed

    def calculate_interval_return(self, current_value, base_value):
        """ 安全计算区间收益率，处理除零错误 """
        if base_value == 0:
            return None
        return (current_value - base_value) / base_value

    def calculate_period_return(self, start_date, end_date):
        """
        计算指定时间段的区间收益率
        :param start_date: 起始日期 (str, 格式 YYYYMMDD)
        :param end_date: 结束日期 (str, 格式 YYYYMMDD)
        :return: 区间收益率 (float or None)
        """
        if start_date not in self.parsed_dict or end_date not in self.parsed_dict:
            return None

        base_value = self.parsed_dict[start_date]
        current_value = self.parsed_dict[end_date]

        return self.calculate_interval_return(current_value, base_value)

    def batch_calculate_returns(self):
        """
        批量计算所有相邻日期间的区间收益率
        :return: DataFrame + 字符串格式结果
        """
        sorted_dates = sorted(self.parsed_dict.keys())
        result_data = []
        out_string = []

        for i in range(1, len(sorted_dates)):
            prev_date = sorted_dates[i - 1]
            curr_date = sorted_dates[i]
            prev_value = self.parsed_dict[prev_date]
            curr_value = self.parsed_dict[curr_date]

            interval_return = self.calculate_interval_return(curr_value, prev_value)

            result_data.append({
                "Start Date": prev_date,
                "End Date": curr_date,
                "Interval Return": interval_return
            })

            return_str = f"{interval_return:.4f}" if interval_return is not None else "null"
            out_string.append(f"{prev_date}~{curr_date}:{return_str}")

        df = pd.DataFrame(result_data)
        return df, "|".join(out_string)

    # 计算上季度末的日期
    def get_quarter_days(self, current_date):
        prev_quarter = ((current_date.month - 1) // 3 + 1) - 1
        if prev_quarter == 0:
            prev_quarter = 4
            year = current_date.year - 1
        else:
            year = current_date.year

        quarter_to_month = {1: 3, 2: 6, 3: 9, 4: 12}
        last_month_of_quarter = quarter_to_month[prev_quarter]
        _, last_day_of_quarter = calendar.monthrange(year, last_month_of_quarter)
        quarter_since_begin = datetime.datetime(year, last_month_of_quarter, last_day_of_quarter, 0, 0, 0)

        return quarter_since_begin

    def annualized_return(self, windows=None):
        """
        年化收益率计算
        """
        if windows is None:
            windows = WINDOWS

        input_str = INPUT_STR
        input_list = input_str.split(",")

        parsed = {}
        for i in input_list:
            date_str, value_str = i.split("^")
            parsed[date_str] = float(value_str)

        sorted_dates = sorted(parsed.keys())
        date_objs = [datetime.datetime.strptime(d, "%Y%m%d") for d in sorted_dates]
        value_list = [parsed[d] for d in sorted_dates]

        result_data = []
        out_string = []

        def calculate_rt_year(current_value, base_value, delta_days):
            if delta_days > 0 and base_value != 0:
                return (current_value - base_value) / base_value * (365 / delta_days)
            return None

        for i in range(len(sorted_dates)):
            current_date_str = sorted_dates[i]
            current_date_obj = date_objs[i]
            current_value = parsed[current_date_str]

            row = {"Date": current_date_str}

            # 当前收益率（与前一天比较）
            if i > 0:
                delta_t = (current_date_obj - date_objs[i - 1]).days
                rt_curr = calculate_rt_year(current_value, value_list[i - 1], delta_t)
                row["d_curr"] = rt_curr
            else:
                row["d_curr"] = None

            # 周期年化收益
            for win in windows:
                if win == 0:
                    row[f"d_{win}"] = 0.0
                    continue

                target_start_date = current_date_obj - timedelta(days=win)
                candidates = [(date_objs[j], value_list[j]) for j in range(i) if date_objs[j] >= target_start_date]

                if not candidates:
                    row[f"d_{win}"] = None
                else:
                    candidates.sort(key=lambda x: x[0])  # 按日期升序排序
                    earliest_date, base_value = candidates[0]
                    delta_t = (current_date_obj - earliest_date).days
                    rt_win = calculate_rt_year(current_value, base_value, delta_t)
                    row[f"d_{win}"] = rt_win

            # 上月/上季/上年以来收益
            month_since_begin = current_date_obj.replace(day=1) - timedelta(days=1)
            quarter_since_begin = self.get_quarter_days(current_date=current_date_obj)
            year_since_begin = current_date_obj.replace(month=1, day=1) - timedelta(days=1)

            idx_month = bisect.bisect_left(date_objs, month_since_begin)
            idx_quarter = bisect.bisect_left(date_objs, quarter_since_begin)
            idx_year = bisect.bisect_left(date_objs, year_since_begin)

            def safe_get(lst, index):
                try:
                    return lst[index]
                except IndexError:
                    return None

            row["d_month"] = calculate_rt_year(current_value, safe_get(value_list, idx_month),
                                               (current_date_obj - safe_get(date_objs,
                                                                            idx_month)).days if idx_month < len(
                                                   date_objs) else 0)
            row["d_quarter"] = calculate_rt_year(current_value, safe_get(value_list, idx_quarter),
                                                 (current_date_obj - safe_get(date_objs,
                                                                              idx_quarter)).days if idx_quarter < len(
                                                     date_objs) else 0)
            row["d_year"] = calculate_rt_year(current_value, safe_get(value_list, idx_year),
                                              (current_date_obj - safe_get(date_objs, idx_year)).days if idx_year < len(
                                                  date_objs) else 0)

            result_data.append(row)

        # 构造 DataFrame
        df = pd.DataFrame(result_data)

        # 构造字符串输出
        def format_with_null(val, prefix):
            if val is None:
                return f"{prefix}:null"
            elif isinstance(val, (int, float)):
                return f"{prefix}:{val:.4f}"
            else:
                return f"{prefix}:{val}"

        for _, row in df.iterrows():
            formatted = [
                format_with_null(row[col], col) for col in df.columns if col != "Date"
            ]
            out_string.append(f"{row['Date']}=>{';'.join(formatted)}")

        return df, "|".join(out_string)

    def valuation_count(self, windows=None):
        """
        估值次数计算
        """
        if windows is None:
            windows = WINDOWS  # 示例窗口期，可根据实际情况调整

        input_str = INPUT_STR  # 示例输入字符串，请替换为实际的INPUT_STR
        input_list = input_str.split(",")

        parsed = {}
        for i in input_list:
            date_str, value_str = i.split("^")
            parsed[date_str] = float(value_str)

        sorted_dates = sorted(parsed.keys())
        date_objs = [datetime.datetime.strptime(d, "%Y%m%d") for d in sorted_dates]
        value_list = [parsed[d] for d in sorted_dates]

        result_data = []
        out_string = []

        for i in range(len(sorted_dates)):
            current_date_str = sorted_dates[i]
            current_date_obj = date_objs[i]
            row = {"Date": current_date_str}

            result_list = []

            for win in windows:
                if win == 0:
                    count = i + 1
                    row[f"d_all"] = count
                    result_list.append(f"d_all:{count}")
                    continue

                target_start_date = current_date_obj - timedelta(days=win)
                candidates = [(date_objs[j], value_list[j]) for j in range(i + 1) if date_objs[j] >= target_start_date]

                count = len(candidates)
                row[f"d_{win}"] = count
                result_list.append(f"d_{win}:{count}")

            # 处理月/季度/年
            month_since_begin = current_date_obj.replace(day=1) - timedelta(days=1)
            quarter_since_begin = self.get_quarter_days(current_date=current_date_obj)
            year_since_begin = current_date_obj.replace(month=1, day=1) - timedelta(days=1)

            idx_month = bisect.bisect_left(date_objs, month_since_begin)
            idx_quarter = bisect.bisect_left(date_objs, quarter_since_begin)
            idx_year = bisect.bisect_left(date_objs, year_since_begin)

            def safe_get(lst, index):
                try:
                    return lst[index]
                except IndexError:
                    return None

            count_month = len(date_objs[idx_month:i + 1])
            count_quarter = len(date_objs[idx_quarter:i + 1])
            count_year = len(date_objs[idx_year:i + 1])

            row["d_month"] = count_month
            row["d_quarter"] = count_quarter
            row["d_year"] = count_year

            result_list.append(f"d_month:{count_month}")
            result_list.append(f"d_quarter:{count_quarter}")
            result_list.append(f"d_year:{count_year}")

            result_data.append(row)
            out_string.append(f"{current_date_str}=>{';'.join(result_list)}")

        df = pd.DataFrame(result_data)
        return df, "|".join(out_string)

    def annualized_volatility(self):
        # 获取估值次数和字符串结果（这里只用到df）
        valuation_df, _ = self.valuation_count(windows=[30])

        # 获取每日收益率数据
        return_df, _ = self.annualized_return(windows=[30])

        # 合并两个DataFrame，确保按日期对齐
        df_merged = pd.merge(valuation_df[['Date', 'd_30']], return_df[['Date', 'd_curr']], on='Date', how='left')
        df_merged.rename(columns={'d_30': 'monthly_count', 'd_curr': 'daily_return'}, inplace=True)
        print(df_merged)

        result_data = []
        out_string = []

        for idx, row in df_merged.iterrows():
            current_date = row['Date']
            monthly_count = row['monthly_count']
            daily_return = row['daily_return']

            volatility_annualized = None

            if pd.notna(monthly_count) and monthly_count >= 12:
                # 收集过去一个月的每日收益率
                window_returns = df_merged.loc[:idx, 'daily_return'].dropna()

                # 检查是否有足够的有效值
                if len(window_returns) >= 12:
                    std_daily = window_returns.std()
                    volatility_annualized = std_daily * (252 ** 0.5)

            result_data.append({
                "Date": current_date,
                "volatility_annualized": volatility_annualized
            })

            # 构建字符串输出格式
            vol_value = f"{volatility_annualized:.4f}" if volatility_annualized is not None else "null"
            out_string.append(f"{current_date}=>volatility_annualized:{vol_value}")

        # 构造DataFrame
        volatility_df = pd.DataFrame(result_data)

        # 返回 DataFrame 和字符串格式
        return volatility_df, "|".join(out_string)


    def run_method(self, method_name):
        if hasattr(self, method_name):
            method = getattr(self, method_name)
            df, str_result = method()
            print("===============================================String result===============================================")
            print(str_result)
            print("===============================================DataFrame result===============================================")
            print(df)
        else:
            print(f"Method {method_name} does not exist.")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Run a specific method of ReturnRiskIndexCalculator")
    parser.add_argument("method", type=str, help="Name of the method to call (e.g., method_a)")
    args = parser.parse_args()
    obj = ReturnRiskIndexCalculator()
    obj.run_method(args.method)

