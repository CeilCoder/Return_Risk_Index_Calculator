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

    def combined_volatility(self, window_days_list=None, min_points=12):
        """
        计算多个窗口下的年化波动率，并返回包含所有结果的DataFrame

        参数:
        - window_days_list: list of int, 多个窗口天数，如 [30, 91]
        - min_points: 最少数据点数量要求

        返回:
        - volatility_df: DataFrame, 每列为对应窗口的波动率
        """
        if window_days_list is None:
            window_days_list = [x for x in WINDOWS if x not in [7, 14, 0]]
        volatility_dict = {window: [] for window in window_days_list}
        date_list = []
        mtd_volatility_list, qtd_volatility_list, ytd_volatility_list = [], [], []

        for i in range(len(self.net_values_series)):
            end_date = self.net_values_series.index[i]
            date_list.append(end_date)

            for window_days in window_days_list:
                vol = self._calculate_volatility(end_date, window_days=window_days, min_points=min_points)
                volatility_dict[window_days].append(vol)

            # 新增：本月以来波动率计算
            mtd_vol = self._calculate_period_volatility(end_date, period_type='month')
            qtd_vol = self._calculate_period_volatility(end_date, period_type='quarter')
            ytd_vol = self._calculate_period_volatility(end_date, period_type='year')
            mtd_volatility_list.append(mtd_vol)
            qtd_volatility_list.append(qtd_vol)
            ytd_volatility_list.append(ytd_vol)

        # 构建 DataFrame
        df_data = {'Date': date_list}
        for window in window_days_list:
            df_data[f'Volatility_{window}D'] = volatility_dict[window]
        df_data['Volatility_MTD'] = mtd_volatility_list  # 新增 MTD 波动率列
        df_data['Volatility_QTD'] = qtd_volatility_list  # 新增 QTD 波动率列
        df_data['Volatility_YTD'] = ytd_volatility_list  # 新增 YTD 波动率列

        volatility_df = pd.DataFrame(df_data)
        return volatility_df

    def _calculate_period_volatility(self, end_date, period_type='month', freq_multiplier=365, min_points=12):
        """
        通用方法：计算从上个月底或本季度初到当前日期的年化波动率

        参数:
        - end_date: pd.Timestamp，当前计算波动率的截止日期
        - period_type: str，支持 'month' 或 'quarter'
        - freq_multiplier: 年化频率，默认为日频（365）

        返回:
        - volatility: float，年化波动率值（保留4位小数），失败则返回 None
        """
        series = self.net_values_series[:end_date]
        if len(series) < 2:
            return None

        # 动态确定周期起始日期
        if period_type == 'month':
            # 上个月最后一天：本月1号减去1天
            current_month_start = pd.Timestamp(end_date.year, end_date.month, 1)
            start_date = current_month_start - pd.Timedelta(days=1)
        elif period_type == 'quarter':
            # 本季度第一天
            quarter_start_month = ((end_date.month - 1) // 3) * 3 + 1
            current_quarter_start = pd.Timestamp(end_date.year, quarter_start_month, 1)
            start_date = current_quarter_start - pd.Timedelta(days=1)
        elif period_type == 'year':
            current_year_start = pd.Timestamp(end_date.year, 1, 1)
            start_date = current_year_start - pd.Timedelta(days=1)
        else:
            raise ValueError("period_type 必须是 'month' 或 'quarter' 或 'year'")

        # 尝试用 get_indexer 寻找最接近但不早于 start_date 的索引
        indexer = series.index.get_indexer([start_date], method='ffill')

        if indexer[0] == -1:
            valid_dates = series.index[series.index >= start_date]
            if len(valid_dates) > 0:
                start_idx = series.index.get_loc(valid_dates[0])  # 取第一个不早于 start_date 的交易日
            else:
                return None
        else:
            start_idx = indexer[0]

        filtered = series.iloc[start_idx:]

        min_point_requirements = {
            'month': 12,  # 对于近1月的数据，估值次数小于12次不计算
            'quarter': 12,  # 对于近3月的数据，估值次数小于12次不计算
            'year': 12,  # 对于近1年的数据，估值次数小于12次不计算
        }
        if len(filtered) < min_point_requirements.get(period_type, min_points):
            return None

        returns = filtered.pct_change().dropna()
        date_diffs = pd.Series(filtered.index).diff().dt.days.replace(0, np.nan).fillna(1)

        conditions = {
            # 30: [(12, freq_multiplier)],
            'quarter': [(36, freq_multiplier), (12, 52)],
            'year': [(144, freq_multiplier), (48, 52), (12, 12)]
        }

        # 计算日化、周化、月化收益率时的系数
        divisors = {
            freq_multiplier: 365,  # 日化
            52: 7,  # 周化
            12: 30  # 月化
        }

        r_t_period = returns / date_diffs.iloc[1:].values
        freq = freq_multiplier

        if period_type in conditions:
            for threshold, frequency in conditions[period_type]:
                if len(filtered) >= threshold:
                    divisors = divisors[frequency]
                    r_t_period = returns / (date_diffs.iloc[1:].values / divisors)
                    freq = frequency
                    break
        else:  # 默认情况下使用日回报率
            r_t_period = returns / date_diffs.iloc[1:].values
            freq = freq_multiplier

        r_period_bar = r_t_period.mean()
        volatility = np.sqrt(((r_t_period - r_period_bar) ** 2).sum() / (len(r_t_period) - 1)) * np.sqrt(freq)

        return round(volatility, 4)


    def _calculate_volatility(self, end_date, window_days=30, min_points=12, freq_multiplier=365):
        i = self.net_values_series.index.get_loc(end_date)
        start_index = max(0, i - window_days)
        filtered = self.net_values_series.iloc[start_index:i + 1]

        min_point_requirements = {
            30: 12,                 # 对于近1月的数据，估值次数小于12次不计算
            91: 12,                 # 对于近3月的数据，估值次数小于12次不计算
            182: 6,                 # 对于近6月的数据，估值次数小于6次不计算
            365: 12,                # 对于近1年的数据，估值次数小于12次不计算
            730: 24,                # 对于近2年的数据，估值次数小于24次不计算
            1095: 36                # 对于近3年的数据，估值次数小于36次不计算
        }

        if len(filtered) < min_point_requirements.get(window_days, min_points):
            return None

        returns = filtered.pct_change().dropna()
        date_diffs = pd.Series(filtered.index).diff().dt.days.replace(0, np.nan).fillna(1)

        # 根据窗口大小选择合适的频率调整方法
        """
        近1月年化波动率：如果近1月估值次数大于等于12次，按照日化收益率计算
        近3月年化波动率：如果近3月估值次数大于等于36次，按照日化收益率计算；如果小于36次大于等于12次，按照周化收益率计算
        近6月年化波动率：如果近6月估值次数大于等于72次，按照日化收益率计算；如果小于72次大于等于24次，按照周化收益率计算；如果小于24次大于等于6次，按照月化收益率计算
        近1年年化波动率：如果近1年估值次数大于等于144次，按照日化收益率计算；如果小于144次大于等于48次，按照周化收益率计算；如果小于48次大于等于12次，按照月化收益率计算
        近2年年化波动率：如果近2年估值次数大于等于288次，按照日化收益率计算；如果小于288次大于等于96次，按照周化收益率计算；如果小于96次大于等于24次，按照月化收益率计算
        近3年年化波动率：如果近3年估值次数大于等于432次，按照日化收益率计算；如果小于432次大于等于144次，按照周化收益率计算；如果小于144次大于等于36次，按照月化收益率计算
        """
        conditions = {
            # 30: [(12, freq_multiplier)],
            91: [(36, freq_multiplier), (12, 52)],
            182: [(72, freq_multiplier), (24, 52), (6, 12)],
            365: [(144, freq_multiplier), (48, 52), (12, 12)],
            730: [(288, freq_multiplier), (96, 52), (24, 12)],
            1095: [(432, freq_multiplier), [144, 52], (36, 12)]
        }

        # 计算日化、周化、月化收益率时的系数
        divisors = {
            freq_multiplier: 365,  # 日化
            52: 7,  # 周化
            12: 30  # 月化
        }

        r_t_period = returns / date_diffs.iloc[1:].values
        freq = freq_multiplier

        if window_days in conditions:
            for threshold, frequency in conditions[window_days]:
                if len(filtered) >= threshold:
                    divisors = divisors[frequency]
                    r_t_period = returns / (date_diffs.iloc[1:].values / divisors)
                    freq = frequency
                    break
        else:  # 默认情况下使用日回报率
            r_t_period = returns / date_diffs.iloc[1:].values
            freq = freq_multiplier

        r_period_bar = r_t_period.mean()
        volatility = np.sqrt(((r_t_period - r_period_bar) ** 2).sum() / (len(r_t_period) - 1)) * np.sqrt(freq)
        return round(volatility, 4)

    # ----------- 夏普比率计算相关方法 --------------

    def annualized_sharpe_ratio(self):
        return 0