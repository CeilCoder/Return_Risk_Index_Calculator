# _*_ coding: utf-8 _*_
"""
@Time : 2025/5/7 15:40
@Auth : Derek
@File : indexCalculate.py
@IDE  : PyCharm
"""

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

INPUT_STR = "20250309^1,20250310^3,20250311^3,20250312^9,20250313^7,20250314^4"

class ReturnRiskIndexCalculator:

    def current_annualized_return_rate(self):
        """
        计算本期年化收益率 / 测试拉取代码
        """
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

            delta_t = None
            rt = None
            rt_year = None

            if pre_value is not None and pre_date is not None:
                delta_t = (current_date_obj - pre_date).days
                if delta_t > 0:
                    rt = (current_value - pre_value) / pre_value
                    rt_year = rt * (365 / delta_t)

            if rt_year is not None:
                out_string.append(f"{current_date_str}=>{rt_year:.4f}")
            else:
                out_string.append(f"{current_date_str}=>null")

            pre_value = current_value
            pre_date = current_date_obj

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

