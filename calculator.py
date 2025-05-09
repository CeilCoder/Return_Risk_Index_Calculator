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

    def combined_volatility(self, window_days_list=[30, 91], min_points=12):
        """
        计算多个窗口下的年化波动率，并返回包含所有结果的DataFrame

        参数:
        - window_days_list: list of int, 多个窗口天数，如 [30, 91]
        - min_points: 最少数据点数量要求

        返回:
        - volatility_df: DataFrame, 每列为对应窗口的波动率
        """
        volatility_dict = {window: [] for window in window_days_list}
        date_list = []

        for i in range(len(self.net_values_series)):
            end_date = self.net_values_series.index[i]
            date_list.append(end_date)

            for window_days in window_days_list:
                vol = self._calculate_volatility(end_date, window_days=window_days, min_points=min_points)
                volatility_dict[window_days].append(vol)

        # 构建 DataFrame
        df_data = {'Date': date_list}
        for window in window_days_list:
            df_data[f'Volatility_{window}D'] = volatility_dict[window]

        volatility_df = pd.DataFrame(df_data)
        return volatility_df


    def _calculate_volatility(self, end_date, window_days=30, min_points=12, freq_multiplier=365):
        i = self.net_values_series.index.get_loc(end_date)
        start_index = max(0, i - window_days)
        filtered = self.net_values_series.iloc[start_index:i + 1]

        if len(filtered) < min_points:
            return None

        returns = filtered.pct_change().dropna()
        date_diffs = pd.Series(filtered.index).diff().dt.days.replace(0, np.nan).fillna(1)

        # 根据窗口大小选择合适的频率调整方法
        if window_days == 91 and len(filtered) >= 36:  # 对于3个月窗口，如果数据点足够，则使用日回报率
            r_t_period = returns / date_diffs.iloc[1:].values
            freq = freq_multiplier
        elif window_days == 91 and 12 <= len(filtered) < 36:  # 如果数据点不够，则尝试用周回报率
            r_t_period = returns / (date_diffs.iloc[1:].values / 7)
            freq = 52
        else:  # 默认情况下使用日回报率
            r_t_period = returns / date_diffs.iloc[1:].values
            freq = freq_multiplier

        r_period_bar = r_t_period.mean()
        volatility = np.sqrt(((r_t_period - r_period_bar) ** 2).sum() / (len(r_t_period) - 1)) * np.sqrt(freq)
        return round(volatility, 4)