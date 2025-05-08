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

    # 计算区间收益率
    def calculate_interval_returns(self, current_value, base_value):
        return (current_value - base_value) / base_value

    def annualized_return(self, windows=None):
        """
        计算 周期年化收益率 和 周期以来的年化收益率
        """

        # 年化收益率计算公式
        def calculate_rt_year(current_value, base_value, delta_days):
            if delta_days > 0:
                return ReturnRiskIndexCalculator.calculate_interval_returns(self, current_value, base_value) * (365 / delta_days)
            return None

        # 空值处理
        def format_with_null(value, prefix):
            if value is not None:
                return f"{prefix}:{value:.4f}"
            else:
                return f"{prefix}:null"

        if windows is None:
            windows = WINDOWS
        input_str = INPUT_STR
        input_list = input_str.split(",")
        # 解析为(date_str, value)的格式，并转换为字典方便查找
        parsed = {}
        for i in input_list:
            date_str, value_str = i.split("^")
            parsed[date_str] = float(value_str)

        # 按日期排序
        sorted_dates = sorted(parsed.keys())
        date_objs = [datetime.datetime.strptime(d, "%Y%m%d") for d in sorted_dates]
        value_list = [parsed[d] for d in sorted_dates]

        # 本期年化收益率 --> 计算 Rt 区间收益率, delta_t 相差天数, Rt_year 本期年化收益率
        result, out_string = [], []
        pre_value = None
        pre_date = None
        for i in range(len(sorted_dates)):
            current_date_str = sorted_dates[i]
            current_date_obj = date_objs[i]
            current_value = parsed[current_date_str]

            result_list = []

            if pre_value is not None and pre_date is not None:
                delta_t = (current_date_obj - pre_date).days
                if delta_t > 0:
                    rt_year_curr = calculate_rt_year(current_value, pre_value, delta_t)
                    result_list.append(format_with_null(rt_year_curr, "d_curr"))
            else:
                result_list.append(format_with_null(None, "d_curr"))

            pre_value = current_value
            pre_date = current_date_obj

        # 周期年化收益率 --> 计算 Rt 区间收益率, delta_t 相差天数, Rt_year 本期年化收益率
            for win in windows:
                # 成立以来的年化收益率需要根据逻辑修改
                if win == 0:
                    result_list.append(f"d_{win}:{0.0}")
                    continue

                target_start_date = current_date_obj - timedelta(days=win)
                # 找出小于等于 current_date，大于等于 target_start_date 的所有日期中最小的那个
                candidates = []
                for j in range(i):
                    if current_date_obj > date_objs[j] >= target_start_date:
                        candidates.append((date_objs[j], value_list[j]))

                if not candidates:
                    result_list.append(f"d_{win}:null")

                # 根据周期算出周期年化收益率
                else:
                    earliest_date = sorted(candidates, key=lambda x: x[0])
                    delta_t = (current_date_obj - earliest_date[0][0]).days
                    fnv_t_0 = earliest_date[0][1]
                    rt_year_win = calculate_rt_year(current_value, fnv_t_0, delta_t)
                    result_list.append(format_with_null(rt_year_win, f"d_{win}"))


        # 周期以来的年化收益率 --> 计算 Rt 区间收益率, delta_t 相差天数, Rt_year 本期年化收益率

            # 处理日期，取上月末、上季度末、上年末
            # 上月末
            month_since_begin = current_date_obj.replace(day=1) - timedelta(days=1)
            # 上季度末
            quarter_since_begin = ReturnRiskIndexCalculator.get_quarter_days(self, current_date=current_date_obj)
            # 上年末
            year_since_begin = current_date_obj.replace(day=1, month=1) - timedelta(days=1)

            # 二分查找，查找当前区间最小日期到当前的取值
            idx_month = bisect.bisect_left(date_objs, month_since_begin)
            idx_quarter = bisect.bisect_left(date_objs, quarter_since_begin)
            idx_year = bisect.bisect_left(date_objs, year_since_begin)

            delta_t_month = (current_date_obj - date_objs[idx_month]).days
            delta_t_quarter = (current_date_obj - date_objs[idx_quarter]).days
            delta_t_year = (current_date_obj - date_objs[idx_year]).days

            rt_year_monthly = calculate_rt_year(current_value, value_list[idx_month], delta_t_month)
            rt_year_quarter = calculate_rt_year(current_value, value_list[idx_quarter], delta_t_quarter)
            rt_year_yearly = calculate_rt_year(current_value, value_list[idx_year], delta_t_year)

            result_list.append(format_with_null(rt_year_monthly, "d_month"))
            result_list.append(format_with_null(rt_year_quarter, "d_quarter"))
            result_list.append(format_with_null(rt_year_yearly, "d_year"))

            result_string = ";".join(result_list)
            out_string.append(f"{current_date_str}=>{result_string}")


        print("|".join(out_string))
        return "|".join(out_string)


    def valuation_count(self, windows=None):
        """
        估值次数计算
        """

        if windows is None:
            windows = WINDOWS
        input_str = INPUT_STR
        input_list = input_str.split(",")
        parsed = {}
        for i in input_list:
            date_str, value_str = i.split("^")
            parsed[date_str] = float(value_str)
        # 按日期排序
        sorted_dates = sorted(parsed.keys())
        date_objs = [datetime.datetime.strptime(d, "%Y%m%d") for d in sorted_dates]

        result, out_string = [], []

        for i in range(1, (len(sorted_dates) + 1)):
            current_date_str = sorted_dates[i - 1]
            current_date_obj = date_objs[i - 1]

            result_list = []

            for win in windows:
                if win == 0:
                    value_counts = len(date_objs[:i])
                    result_list.append(f"d_all:{value_counts}")
                    continue
                target_start_date = current_date_obj - timedelta(days=win)
                candidates = [v for v in date_objs[:i] if target_start_date <= v <= current_date_obj]
                if not candidates:
                    result_list.append(f"d_{win}:null")
                else:
                    value_counts = len(candidates)
                    result_list.append(f"d_{win}:{value_counts}")

            # 处理日期，取上月末、上季度末、上年末
            # 上月末
            month_since_begin = current_date_obj.replace(day=1) - timedelta(days=1)
            # 上季度末
            quarter_since_begin = ReturnRiskIndexCalculator.get_quarter_days(self, current_date=current_date_obj)
            # 上年末
            year_since_begin = current_date_obj.replace(day=1, month=1) - timedelta(days=1)

            idx_month = bisect.bisect_left(date_objs, month_since_begin)
            idx_quarter = bisect.bisect_left(date_objs, quarter_since_begin)
            idx_year = bisect.bisect_left(date_objs, year_since_begin)
            idx_current = bisect.bisect_left(date_objs, current_date_obj)
            value_counts_month = len(date_objs[idx_month:idx_current + 1])
            value_counts_quarter = len(date_objs[idx_quarter:idx_current + 1])
            value_counts_year = len(date_objs[idx_year:idx_current + 1])
            result_list.append(f"d_month:{value_counts_month}")
            result_list.append(f"d_quarter:{value_counts_quarter}")
            result_list.append(f"d_year:{value_counts_year}")
            # current是当前日期，date_obj[idx_month]是列表中离上月末最近的日期，month_since_begin是上月末
            # print(current_date_obj, value_counts_month, value_counts_quarter, value_counts_year)



            result_string = ";".join(result_list)
            out_string.append(f"{current_date_str}=>{result_string}")

        print("|".join(out_string))
        return "|".join(out_string)


    def annualized_volatility(self, windows=None):
        """
        计算年化波动率
        """
        if windows is None:
            windows = WINDOWS
        windows = [30]
        t_windows = [365]



    def run_method(self, method_name):
        if hasattr(self, method_name):
            method = getattr(self, method_name)
            method()
        else:
            print(f"Method {method_name} does not exist.")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Run a specific method of ReturnRiskIndexCalculator")
    parser.add_argument("method", type=str, help="Name of the method to call (e.g., method_a)")
    args = parser.parse_args()
    obj = ReturnRiskIndexCalculator()
    obj.run_method(args.method)

