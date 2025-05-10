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

    def annualized_volatility(self, window_days_list=None, min_points=12):
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
        period_list = ['month', 'quarter', 'year']

        # 合并周期列表
        periods_to_calculate = {f'Volatility_{win}D': win for win in window_days_list}
        periods_to_calculate.update({f'Volatility_{period}D': period for period in period_list})

        volatility_data = {key: [] for key in ['Date'] + list(periods_to_calculate.keys())}

        for i in range(len(self.net_values_series)):
            end_date = self.net_values_series.index[i]
            volatility_data['Date'].append(end_date)

            for name, period in periods_to_calculate.items():
                if isinstance(period, int):  # 如果是天数窗口
                    monthly_returns, r_t_day = self._calculate_volatility(end_date=end_date, window_days=period)
                else:  # 如果是时间段
                    monthly_returns, r_t_day = self._calculate_volatility(end_date=end_date, period_type=period)

                volatility_data[name].append(monthly_returns)

        volatility_df = pd.DataFrame(volatility_data)
        return volatility_df

    def _calculate_volatility(self, end_date, period_type=None, window_days=None, freq_multiplier=365, min_points=12):
        series = self.net_values_series[:end_date]
        if len(series) < 2:
            return None, None

        # 确定起始日期
        if period_type:
            if period_type == 'month':
                start_date = pd.Timestamp(end_date.year, end_date.month, 1) - pd.Timedelta(days=1)
            elif period_type == 'quarter':
                q_start_month = ((end_date.month - 1) // 3) * 3 + 1
                start_date = pd.Timestamp(end_date.year, q_start_month, 1) - pd.Timedelta(days=1)
            elif period_type == 'year':
                start_date = pd.Timestamp(end_date.year, 1, 1) - pd.Timedelta(days=1)
            else:
                raise ValueError("Unsupported period_type")

            indexer = series.index.get_indexer([start_date], method='ffill')
            start_idx = indexer[0] if indexer[0] != -1 else next(
                (i for i, d in enumerate(series.index) if d >= start_date), None)
            if start_idx is None:
                return None, None
            filtered = series.iloc[start_idx:]

        elif window_days is not None:
            i = series.index.get_loc(end_date)
            start_index = max(0, i - window_days)
            filtered = series.iloc[start_index:i + 1]
        else:
            raise ValueError("Must specify either period_type or window_days")

        # 检查数据量是否满足最低要求
        key = period_type if period_type else window_days
        min_required = {
            'month': 12,
            'quarter': 12,
            'year': 12,
            30: 12,                 # 对于近1月的数据，估值次数小于12次不计算
            91: 12,                 # 对于近3月的数据，估值次数小于12次不计算
            182: 6,                 # 对于近6月的数据，估值次数小于6次不计算
            365: 12,                # 对于近1年的数据，估值次数小于12次不计算
            730: 24,                # 对于近2年的数据，估值次数小于24次不计算
            1095: 36                # 对于近3年的数据，估值次数小于36次不计算
        }.get(key, min_points)

        if len(filtered) < min_required:
            return None, None

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
            'month': [(12, freq_multiplier)],
            'quarter': [(36, freq_multiplier), (12, 52)],
            'year': [(144, freq_multiplier), (48, 52), (12, 12)],
            91: [(36, freq_multiplier), (12, 52)],
            182: [(72, freq_multiplier), (24, 52), (6, 12)],
            365: [(144, freq_multiplier), (48, 52), (12, 12)],
            730: [(288, freq_multiplier), (96, 52), (24, 12)],
            1095: [(432, freq_multiplier), (144, 52), (36, 12)]
        }

        # 计算日化、周化、月化收益率时的系数
        divisors = {
            freq_multiplier: 365,
            52: 7,
            12: 30
        }

        freq = freq_multiplier
        r_t_period = returns / date_diffs.iloc[1:].values

        if key in conditions:
            for threshold, frequency in conditions[key]:
                if len(filtered) >= threshold:
                    r_t_period = returns / (date_diffs.iloc[1:].values / divisors[frequency])
                    freq = frequency

        r_period_bar = r_t_period.mean()
        volatility = np.sqrt(((r_t_period - r_period_bar) ** 2).sum() / (len(r_t_period) - 1)) * np.sqrt(freq)
        return round(volatility, 4), r_t_period

    # ----------- 夏普比率计算相关方法 --------------

    def annualized_sharpe_ratio(self, window_days_list=None):
        """
        优化后的计算多个窗口下的夏普比率的方法
        :return: DataFrame：包含每个时间节点的夏普比率
        """
        if window_days_list is None:
            window_days_list = [x for x in WINDOWS if x not in [7, 14, 0]]
        period_list = ['month', 'quarter', 'year']

        # 合并周期列表
        periods_to_calculate = {f'Sharpe_{win}D': win for win in window_days_list}
        periods_to_calculate.update({f'Sharpe_{period}D': period for period in period_list})

        sharpe_data = {key: [] for key in ['Date'] + list(periods_to_calculate.keys())}

        for i in range(len(self.net_values_series)):
            end_date = self.net_values_series.index[i]
            sharpe_data['Date'].append(end_date)

            for name, period in periods_to_calculate.items():
                if isinstance(period, int):  # 如果是天数窗口
                    monthly_returns, r_t_day = self._calculate_volatility(end_date=end_date, window_days=period)
                else:  # 如果是时间段
                    monthly_returns, r_t_day = self._calculate_volatility(end_date=end_date, period_type=period)

                if monthly_returns is not None and r_t_day is not None:
                    r_t_year_day = (r_t_day * 365).mean()
                    sharpe_ratio = round((r_t_year_day - 0.015) / monthly_returns, 4)
                else:
                    sharpe_ratio = None

                sharpe_data[name].append(sharpe_ratio)

        sharpe_df = pd.DataFrame(sharpe_data)
        return sharpe_df

    # ----------- 最大回撤计算相关方法 --------------

    def max_drawdown(self, window=None):
        """
        优化后的计算多个窗口下的最大回撤的方法
        :return: DataFrame：包含每个时间节点的最大回撤
        """
        if window is None:
            window_list = [x for x in WINDOWS if x not in [0]]
        period_list = ['month', 'quarter', 'year']

        # 合并周期列表
        periods_to_calculate = {f'Max_drawdown_{win}D': win for win in window_list}
        periods_to_calculate.update({f'Max_drawdown_{period}D': period for period in period_list})

        max_drawdown_data = {key: [] for key in ['Date'] + list(periods_to_calculate.keys())}

        for i in range(len(self.net_values_series)):
            end_date = self.net_values_series.index[i]
            max_drawdown_data['Date'].append(end_date)

            for name, period in periods_to_calculate.items():
                if isinstance(period, int):  # 如果是天数窗口
                    max_drawdown, s_date, t_date, d_r_d = self._calculate_max_drawdown(end_date=end_date, windows=period)
                else:  # 如果是时间段
                    max_drawdown, s_date, t_date, d_r_d = self._calculate_max_drawdown(end_date=end_date, period_type=period)

                max_drawdown_data[name].append(max_drawdown)

        max_drawdown_df = pd.DataFrame(max_drawdown_data)
        return max_drawdown_df

    def _calculate_max_drawdown(self, end_date, period_type=None, windows=None):
        # 确定起始日期
        i = self.net_values_series.index.get_loc(end_date)
        if period_type:
            if period_type == 'month':
                start_date = pd.Timestamp(end_date.year, end_date.month, 1) - pd.Timedelta(days=1)
            elif period_type == 'quarter':
                q_start_month = ((end_date.month - 1) // 3) * 3 + 1
                start_date = pd.Timestamp(end_date.year, q_start_month, 1) - pd.Timedelta(days=1)
            elif period_type == 'year':
                start_date = pd.Timestamp(end_date.year, 1, 1) - pd.Timedelta(days=1)
            else:
                raise ValueError("Unsupported period_type")

            indexer = self.net_values_series.index.get_indexer([start_date], method='ffill')
            start_idx = indexer[0] if indexer[0] != -1 else next((i for i, d in enumerate(self.net_values_series.index) if d >= start_date), None)
            filtered = self.net_values_series.iloc[start_idx: i + 1]
        elif windows is not None:
            start_index = max(0, i - windows)
            filtered = self.net_values_series.iloc[start_index: i + 1]
        else:
            raise ValueError("Must specify either period_type or window_days")


        values = filtered.values
        dates = filtered.index

        peak = values[0]
        peak_end_date = end_date
        peak_start_date = dates[0]
        max_drawdown = 0.0
        drawdown_start, drawdown_end = peak_start_date, peak_end_date
        drawdown_repair_days = None

        for j in range(1, len(values)):
            current_value = values[j]
            current_date = dates[j]

            if current_value > peak:
                peak = current_value
                peak_start_date = current_date
            else:
                drawdown = (current_value - peak) / peak
                if drawdown < max_drawdown:
                    max_drawdown = drawdown
                    drawdown_start = peak_start_date
                    drawdown_end = current_date
                    drawdown_repair_days = (drawdown_end - drawdown_start).days

        # 处理格式

        result_str = f"{max_drawdown:.4f}(S:{drawdown_start.strftime('%Y%m%d')},T:{drawdown_end.strftime('%Y%m%d')},DRD:{drawdown_repair_days})"

        return round(max_drawdown, 4), drawdown_start.strftime('%Y%m%d'), drawdown_end.strftime('%Y%m%d'), drawdown_repair_days

    # ----------- 卡玛比率计算相关方法 --------------

    def annualized_calmer_ratio(self, windows=None):
        """
        优化后的计算多个窗口下的卡玛比率的方法
        :return: DataFrame：包含每个时间节点的卡玛比率
        """
        if windows is None:
            window_days_list = [x for x in WINDOWS if x not in [0]]
        period_list = ['month', 'quarter', 'year']

        # 合并周期列表
        periods_to_calculate = {f'Calmer_{win}D': win for win in window_days_list}
        periods_to_calculate.update({f'Calmer_{period}D': period for period in period_list})

        calmer_data = {key: [] for key in ['Date'] + list(periods_to_calculate.keys())}

        for i in range(len(self.net_values_series)):
            end_date = self.net_values_series.index[i]
            calmer_data['Date'].append(end_date)

            for name, period in periods_to_calculate.items():
                if isinstance(period, int):  # 如果是天数窗口
                    max_drawdown, s_date, t_date, d_r_d = self._calculate_max_drawdown(end_date=end_date, windows=period)
                    # returns = self.annualized_return().loc(end_date, f"d_{period}")
                else:  # 如果是时间段
                    # returns = self.annualized_return().loc(end_date, f"d_{period}")
                    max_drawdown, s_date, t_date, d_r_d = self._calculate_max_drawdown(end_date=end_date, period_type=period)

                returns = self.annualized_return()

                if max_drawdown is not None and returns is not None:
                    calmer_ratio = returns
                    # calmer_ratio = returns / abs(min(max_drawdown, 0))
                else:
                    calmer_ratio = None

                calmer_data[name].append(calmer_ratio)

        calmer_df = pd.DataFrame(calmer_data)
        return calmer_df



