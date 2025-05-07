# _*_ coding: utf-8 _*_
"""
@Time : 2025/5/7 15:40
@Auth : Derek
@File : indexCalculate.py
@IDE  : PyCharm
"""
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

INPUT_STR = "20250309^1,20250310^3,20250311^3,20250312^9,20250313^7,20250314^4,20250315^1,20250316^1,20250317^7,20250318^3,20250319^7,20250320^3,20250321^8,20250322^7,20250323^1,20250324^3,20250325^2,20250326^8,20250327^9,20250328^1,20250329^1,20250330^3,20250331^7,20250401^6"
WINDOWS = [7, 14, 30, 90, 182, 365, 730, 1095, 0] # 窗口周期

class ReturnRiskIndexCalculator:

    def current_annualized_return_rate(self, windows=None):
        """
        计算本期年化收益率 / 测试从GitHub拉取代码
        """
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

        # 计算 Rt 区间收益率, delta_t 相差天数, Rt_year 本期年化收益率
        result, out_string = [], []
        pre_value = None
        pre_date = None
        for i in range(len(sorted_dates)):
            current_date_str = sorted_dates[i]
            current_date_obj = date_objs[i]
            current_value = parsed[current_date_str]

            rt_year = None

            if pre_value is not None and pre_date is not None:
                delta_t = (current_date_obj - pre_date).days
                if delta_t > 0:
                    rt_year = ((current_value - pre_value) / pre_value) * (365 / delta_t)

            if rt_year is not None:
                out_string.append(f"{current_date_str}=>d_curr:{rt_year:.4f}")
            else:
                out_string.append(f"{current_date_str}=>null")

            pre_value = current_value
            pre_date = current_date_obj

        print("|".join(out_string))
        return "|".join(out_string)

    def period_annualized_return(self, windows=None):
        """
        计算周期年化收益率
        """
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

        # 计算 Rt 区间收益率, delta_t 相差天数, Rt_year 本期年化收益率
        result, out_string = [], []
        pre_value = None
        pre_date = None
        for i in range(len(sorted_dates)):
            current_date_str = sorted_dates[i]
            current_date_obj = date_objs[i]
            current_value = parsed[current_date_str]

            result_list = []

            for win in windows:
                # 全历史需要根据逻辑修改
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
                    fnv_t_0 = earliest_date[0][1]
                    rt_1 = (current_value - fnv_t_0) / fnv_t_0
                    result_list.append(f"d_{win}:{rt_1}")

            result_string = ";".join(result_list)
            out_string.append(f"{current_date_str}=>{result_string}")
        print("|".join(out_string))
        return "|".join(out_string)


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

