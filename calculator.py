# return_calculator/calculator.py

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import bisect
from utils import calculate_interval_return, annualized_return, get_quarter_days
from config import WINDOWS


class ReturnRiskIndexCalculator:
    def __init__(self, net_values_series):
        self.net_values_series = net_values_series
        self.date_objs = net_values_series.index.to_pydatetime().tolist()
        self.value_list = net_values_series.tolist()

    # ----------- 收益率计算相关方法 --------------

    def batch_calculate_returns(self):
        """计算区间收益率"""
        returns = self.net_values_series.pct_change()
        return returns.dropna()  # 删除第一个NaN值

    def annualized_return(self):
        """计算年化收益率"""
        result_data = []
        for dt in self.net_values_series.index:
            row = {"Date": dt.strftime("%Y%m%d")}
            # 当前与前一天比较
            if dt != self.net_values_series.index[0]:
                prev_dt = self.net_values_series.index[self.net_values_series.index.get_loc(dt) - 1]
                interval_return = calculate_interval_return(self.net_values_series.loc[dt], self.net_values_series.loc[prev_dt])
                row["d_curr"] = annualized_return(interval_return, (dt - prev_dt).days)

            # 各窗口期年化收益
            for win in WINDOWS:
                if win == 0:
                    row[f"d_{win}"] = 0.0
                    continue

                start_dt = dt - timedelta(days=win)
                try:
                    base_value = self.net_values_series[start_dt:].iloc[0]
                    interval_return = calculate_interval_return(self.net_values_series.loc[dt], base_value)
                    row[f"d_{win}"] = annualized_return(interval_return, (dt - start_dt).days)
                except IndexError:
                    row[f"d_{win}"] = None

            # 月、季、年
            month_start = dt.replace(day=1) - timedelta(days=1)
            quarter_start = get_quarter_days(dt)
            year_start = dt.replace(month=1, day=1) - timedelta(days=1)

            periods = [("month", month_start), ("quarter", quarter_start), ("year", year_start)]
            for period, start in periods:
                try:
                    base_value = self.net_values_series[start:].iloc[0]
                    interval_return = calculate_interval_return(self.net_values_series.loc[dt], base_value)
                    row[f"d_{period}"] = annualized_return(interval_return, (dt - start).days)
                except IndexError:
                    row[f"d_{period}"] = None

            result_data.append(row)

        return pd.DataFrame(result_data)

    # ----------- 估值次数计算相关方法 --------------

    def count_valuation(self, windows=None):
        """
        计算每个时间点上不同窗口期内的数据点数量
        """
        if windows is None:
            windows = WINDOWS

        result_data = []
        out_strings = []

        for i in range(len(self.net_values_series)):
            current_date_str = self.date_objs[i].strftime("%Y%m%d")
            current_date_obj = self.date_objs[i]
            row = {"Date": current_date_str}
            result_list = []

            for win in windows:
                if win == 0:
                    count = i + 1
                    row[f"d_all"] = count
                    result_list.append(f"d_all:{count}")
                    continue

                target_start_date = current_date_obj - timedelta(days=win)
                candidates = [(self.date_objs[j], self.value_list[j]) for j in range(i + 1) if self.date_objs[j] >= target_start_date]
                count = len(candidates)
                row[f"d_{win}"] = count
                result_list.append(f"d_{win}:{count}")

            # 处理月度、季度、年度
            month_since_begin = current_date_obj.replace(day=1) - timedelta(days=1)
            quarter_since_begin = get_quarter_days(current_date=current_date_obj)
            year_since_begin = current_date_obj.replace(month=1, day=1) - timedelta(days=1)

            idx_month = bisect.bisect_left(self.date_objs, month_since_begin)
            idx_quarter = bisect.bisect_left(self.date_objs, quarter_since_begin)
            idx_year = bisect.bisect_left(self.date_objs, year_since_begin)

            count_month = len(self.date_objs[idx_month:i + 1])
            count_quarter = len(self.date_objs[idx_quarter:i + 1])
            count_year = len(self.date_objs[idx_year:i + 1])

            row["d_month"] = count_month
            row["d_quarter"] = count_quarter
            row["d_year"] = count_year

            result_list.append(f"d_month:{count_month}")
            result_list.append(f"d_quarter:{count_quarter}")
            result_list.append(f"d_year:{count_year}")

            result_data.append(row)
            out_strings.append(f"{current_date_str}=>{';'.join(result_list)}")

        df = pd.DataFrame(result_data)
        return df, "|".join(out_strings)

    def annualized_volatility(self, windows=None, min_points=12):
        """
        支持多个窗口期（如 30 天、91 天）的年化波动率计算。
        返回一个 DataFrame，包含 'Date' 和各窗口期对应的波动率列。

        参数：
            windows (list): 窗口期配置，格式如：[30, 91]
            min_points (int): 最少估值点数量要求，默认为 12
        """
        if windows is None:
            windows = [30, 91]

        # 初始化结果字典
        volatility_dict = {f'{window}天年化波动率': [] for window in windows}
        date_list = []

        index_dates = self.net_values_series.index

        for i, date in enumerate(index_dates):
            date_list.append(date)
            returns_window = self.net_values_series.iloc[max(0, i - max(windows)):i + 1]

            # 判断估值频率
            annualized_volatility_data_1m = self.net_values_series[
                (self.net_values_series.index > date - timedelta(days=30)) &
                (self.net_values_series.index <= date)
            ]
            num_annualized_volatility_data_1m = len(annualized_volatility_data_1m)

            annualized_volatility_data_3m = self.net_values_series[
                (self.net_values_series.index > date - timedelta(days=91)) &
                (self.net_values_series.index <= date)
            ]
            num_annualized_volatility_data_3m = len(annualized_volatility_data_3m)

            if num_annualized_volatility_data_1m < min_points or num_annualized_volatility_data_3m < min_points:
                for col_name in volatility_dict.keys():
                    volatility_dict[col_name].append(None)
                continue

            # 根据估值次数选择计算方法
            if min_points <= num_annualized_volatility_data_3m < 36:
                freq_multiplier = 52
            elif num_annualized_volatility_data_3m >= 36 or num_annualized_volatility_data_1m >= 12:
                freq_multiplier = 365
            else:
                freq_multiplier = None

            for window_days in windows:
                start_date = date - timedelta(days=window_days)

                # 如果数据起始时间早于窗口开始时间，则截取对应窗口数据
                filtered = self.net_values_series[
                    (self.net_values_series.index > start_date) &
                    (self.net_values_series.index <= date)
                    ]

                if len(filtered) < min_points:
                    volatility_dict[f'{window_days}天年化波动率'].append(None)
                    continue

                returns = filtered.pct_change().dropna()
                date_diffs = pd.Series(filtered.index).diff().dt.days.replace(0, np.nan).fillna(1)
                if freq_multiplier == 52:
                    r_t_week = returns / (date_diffs.values[1:] / 7)
                    r_day_bar = r_t_week.mean()
                    volatility = np.sqrt(((r_t_week - r_day_bar) ** 2).sum() / (len(r_t_week) - 1)) * np.sqrt(freq_multiplier)
                else:
                    r_t_day = returns / date_diffs.iloc[1:].values
                    r_day_bar = r_t_day.mean()
                    volatility = np.sqrt(((r_t_day - r_day_bar) ** 2).sum() / (len(r_t_day) - 1)) * np.sqrt(365)

                volatility = round(volatility, 4)  # 保留4位小数

                volatility_dict[f'{window_days}天年化波动率'].append(volatility)

        # 构建 DataFrame
        result_df = pd.DataFrame({
            'Date': date_list,
            **volatility_dict
        })

        return result_df