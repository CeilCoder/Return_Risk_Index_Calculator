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

logger = logging.getLogger()

class ReturnRiskIndexCalculator(object):
    def __init__(self, returns):
        self.returns = returns

    def current_annualized_return_rate(self):
        """
        计算本期年化收益率
        """
