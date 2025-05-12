import pandas as pd
import numpy as np
from config import WINDOWS


class ReturnRiskIndexCalculator:
    def __init__(self, net_values_series):
        self.net_values_series = net_values_series
        self.date_objs = net_values_series.index.to_pydatetime().tolist()
        self.value_list = net_values_series.tolist()

    # ----------- 起始日期计算相关方法 --------------
    def _get_start_date(self, end_date, period_type=None, windows=None):
        if period_type:
            if period_type == 'month':
                start_date = pd.Timestamp(end_date.year, end_date.month, 1) - pd.Timedelta(days=1)
            elif period_type == 'quarter':
                q_start_month = ((end_date.month - 1) // 3) * 3 + 1
                start_date = pd.Timestamp(end_date.year, q_start_month, 1) - pd.Timedelta(days=1)
            elif period_type == 'year':
                start_date = pd.Timestamp(end_date.year, 1, 1) - pd.Timedelta(days=1)
            elif period_type == 'current':
                start_date = pd.Timestamp(end_date) - pd.Timedelta(days=1)
            else:
                raise ValueError("Unsupported period_type")
        elif windows is not None:
            i = self.net_values_series.index.get_loc(end_date)
            return self.net_values_series.index[max(0, i - windows)]
        else:
            raise ValueError("Must specify either period_type or window_days")
        return start_date

    def _filter_data_by_dates(self, start_date, end_date):
        indexer = self.net_values_series.index.get_indexer([start_date], method='ffill')
        start_idx = indexer[0] if indexer[0] != -1 else next(
            (i for i, d in enumerate(self.net_values_series.index) if d >= start_date), None)
        return self.net_values_series.iloc[start_idx:self.net_values_series.index.get_loc(end_date) + 1]

    # ----------- 收益率计算相关方法 --------------

    def annualized_return(self, windows=None):
        """
        计算多个窗口下的年化收益率，并返回包含所有结果的DataFrame

        参数:
        - window_days_list: list of int, 多个窗口天数，如 [30, 91]

        返回:
        - returns_df: DataFrame, 每列为对应窗口的收益率
        """
        if windows is None:
            window_days_list = [x for x in WINDOWS]
        period_list = ['month', 'quarter', 'year', 'current']

        # 合并周期列表
        periods_to_calculate = {f'Returns_{win}D': win for win in window_days_list}
        periods_to_calculate.update({f'Returns_{period}D': period for period in period_list})

        returns_data = {key: [] for key in ['Date'] + list(periods_to_calculate.keys())}

        for i in range(len(self.net_values_series)):
            end_date = self.net_values_series.index[i]
            returns_data['Date'].append(end_date)

            for name, period in periods_to_calculate.items():
                if isinstance(period, int):  # 如果是天数窗口
                    returns = self._calculate_annualized_returns(end_date=end_date, windows=period)
                else:  # 如果是时间段
                    returns = self._calculate_annualized_returns(end_date=end_date, period_type=period)

                returns_data[name].append(returns)

        returns_df = pd.DataFrame(returns_data)
        return returns_df

    def _calculate_annualized_returns(self, end_date, period_type=None, windows=None, min_points=None):
        # 确定起始日期
        product_start_date = self.net_values_series.index[0]

        if windows == 0:
            start_date = product_start_date
        else:
            start_date = self._get_start_date(end_date, period_type=period_type, windows=windows)

        filtered = self._filter_data_by_dates(start_date=start_date, end_date=end_date)

        # 检查数据量是否满足最低要求
        key = period_type if period_type else windows
        min_required = {
            'month': 1,
            'quarter': 1,
            'year': 1,
            'current': 1,
            0: 1,
            7: 7,
            14: 14,
            30: 30,     # 对于近1月的数据，估值次数小于12次不计算
            91: 91,     # 对于近3月的数据，估值次数小于12次不计算
            182: 182,   # 对于近6月的数据，估值次数小于6次不计算
            365: 365,   # 对于近1年的数据，估值次数小于12次不计算
            730: 730,   # 对于近2年的数据，估值次数小于24次不计算
            1095: 1095  # 对于近3年的数据，估值次数小于36次不计算
        }.get(key, min_points)

        values = filtered.values
        dates = filtered.index

        # 周期内净值的起始日期、起始日期对应的净值、终止日期、终止日期对应的净值、相差天数
        start_date = dates[0]
        start_value = values[0]
        end_value = values[-1]
        days_diff = (end_date - start_date).days

        if len(filtered) <= min_required and days_diff == 0:
            returns_year = None
        else:
            returns = (end_value - start_value) / start_value
            returns_year = returns * (365 / days_diff)
            returns_year = round(returns_year, 4)
        return returns_year

    # ----------- 估值次数计算相关方法 --------------

    def valuation_count(self, windows=None):
        """
        计算多个窗口下的估值次数，并返回包含所有结果的DataFrame

        参数:
        - window_days_list: list of int, 多个窗口天数，如 [30, 91]

        返回:
        - counts_df: DataFrame, 每列为对应窗口的估值次数
        """
        if windows is None:
            window_days_list = [x for x in WINDOWS]
        period_list = ['month', 'quarter', 'year']

        # 合并周期列表
        periods_to_calculate = {f'Count_{win}D': win for win in window_days_list}
        periods_to_calculate.update({f'Count_{period}D': period for period in period_list})

        valuation_count_data = {key: [] for key in ['Date'] + list(periods_to_calculate.keys())}

        for i in range(len(self.net_values_series)):
            end_date = self.net_values_series.index[i]
            valuation_count_data['Date'].append(end_date)

            for name, period in periods_to_calculate.items():
                if isinstance(period, int):  # 如果是天数窗口
                    counts = self._calculate_valuation_count(end_date=end_date, windows=period)
                else:  # 如果是时间段
                    counts = self._calculate_valuation_count(end_date=end_date, period_type=period)

                valuation_count_data[name].append(counts)

        counts_df = pd.DataFrame(valuation_count_data)
        return counts_df

    def _calculate_valuation_count(self, end_date, windows=None, period_type = None, min_points=None):
        # 确定起始日期
        product_start_date = self.net_values_series.index[0]

        if windows == 0:
            start_date = product_start_date
        else:
            start_date = self._get_start_date(end_date, period_type=period_type, windows=windows)

        filtered = self._filter_data_by_dates(start_date=start_date, end_date=end_date)

        # 检查数据量是否满足最低要求
        key = period_type if period_type else windows
        min_required = {
            'month': 1,
            'quarter': 1,
            'year': 1,
            'current': 1,
            0: 1,
            7: 0,
            14: 14,
            30: 30,  # 对于近1月的数据，估值次数小于12次不计算
            91: 91,  # 对于近3月的数据，估值次数小于12次不计算
            182: 182,  # 对于近6月的数据，估值次数小于6次不计算
            365: 365,  # 对于近1年的数据，估值次数小于12次不计算
            730: 730,  # 对于近2年的数据，估值次数小于24次不计算
            1095: 1095  # 对于近3年的数据，估值次数小于36次不计算
        }.get(key, min_points)

        if (end_date - product_start_date).days < 0:
            return None
        else:
            return len(filtered)

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
            window_days_list = [x for x in WINDOWS if x not in [7, 14]]
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
                    monthly_returns, r_t_day = self._calculate_volatility(end_date=end_date, windows=period)
                else:  # 如果是时间段
                    monthly_returns, r_t_day = self._calculate_volatility(end_date=end_date, period_type=period)

                volatility_data[name].append(monthly_returns)

        volatility_df = pd.DataFrame(volatility_data)
        return volatility_df

    def _calculate_volatility(self, end_date, period_type=None, windows=None, freq_multiplier=365, min_points=12):
        series = self.net_values_series[:end_date]
        product_start_date = self.net_values_series.index[0]
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

        elif windows is not None:
            i = series.index.get_loc(end_date)
            if windows == 0:
                filtered = series.iloc[0:i + 1]
            else:
                start_index = max(0, i - windows)
                filtered = series.iloc[start_index:i + 1]
        else:
            raise ValueError("Must specify either period_type or window_days")

        # 检查数据量是否满足最低要求
        key = period_type if period_type else windows
        min_required = {
            'month': 12,
            'quarter': 12,
            'year': 12,
            0: 182,
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

        n = int(len(filtered) / 182)

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
            0: [(72 * n, freq_multiplier), (24 * n, 52), (6 * n, 12)],
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
            window_days_list = [x for x in WINDOWS if x not in [7, 14]]
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
            window_list = [x for x in WINDOWS]
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

                max_drawdown_data[name].append(f"{max_drawdown}(S:{s_date},T:{t_date},DRD:{d_r_d})")

        max_drawdown_df = pd.DataFrame(max_drawdown_data)
        return max_drawdown_df

    def _calculate_max_drawdown(self, end_date, period_type=None, windows=None, min_points=None):
        # 确定起始日期
        product_start_date = self.net_values_series.index[0]
        if windows == 0:
            start_date = product_start_date
        else:
            start_date = self._get_start_date(end_date=end_date, period_type=period_type, windows=windows)

        filtered = self._filter_data_by_dates(start_date=start_date, end_date=end_date)

        # 检查数据量是否满足最低要求
        key = period_type if period_type else windows
        min_required = {
            'month': 1,
            'quarter': 1,
            'year': 1,
            'current': 1,
            0: 1,
            7: 7,
            14: 14,
            30: 30,  # 对于近1月的数据，估值次数小于12次不计算
            91: 91,  # 对于近3月的数据，估值次数小于12次不计算
            182: 182,  # 对于近6月的数据，估值次数小于6次不计算
            365: 365,  # 对于近1年的数据，估值次数小于12次不计算
            730: 730,  # 对于近2年的数据，估值次数小于24次不计算
            1095: 1095  # 对于近3年的数据，估值次数小于36次不计算
        }.get(key, min_points)

        if (end_date - product_start_date).days < min_required:
            return None, None, None, None
        else:


            values = filtered.values
            dates = filtered.index

            peak = values[0]
            peak_end_date = end_date
            peak_start_date = dates[0]
            max_drawdown = 0.0
            drawdown_start, drawdown_end = peak_start_date, peak_end_date
            drawdown_repair_days = (drawdown_end - drawdown_start).days

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

            return round(max_drawdown, 4), drawdown_start.strftime('%Y%m%d'), drawdown_end.strftime('%Y%m%d'), drawdown_repair_days

    # ----------- 卡玛比率计算相关方法 --------------

    def annualized_calmer_ratio(self, windows=None):
        """
        优化后的计算多个窗口下的卡玛比率的方法
        :return: DataFrame：包含每个时间节点的卡玛比率
        """
        if windows is None:
            window_days_list = [x for x in WINDOWS]
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
                    returns = self._calculate_annualized_returns(end_date=end_date, windows=period)
                else:  # 如果是时间段
                    max_drawdown, s_date, t_date, d_r_d = self._calculate_max_drawdown(end_date=end_date, period_type=period)
                    returns = self._calculate_annualized_returns(end_date=end_date, period_type=period)

                if max_drawdown is not None and returns is not None:
                    if abs(min(max_drawdown, 0.0)) == 0.0:
                        calmer_ratio = None
                    else:
                        calmer_ratio = returns / abs(min(max_drawdown, 0.0))
                else:
                    calmer_ratio = None

                calmer_data[name].append(calmer_ratio)

        calmer_df = pd.DataFrame(calmer_data)
        return calmer_df

    # ----------- 回撤计算相关方法 --------------

    def _calculator_drawdown(self, end_date, period_type=None, windows=None, min_points=None):
        # 确定起始日期
        i = self.net_values_series.index.get_loc(end_date)
        product_start_date = self.net_values_series.index[0]

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
            start_idx = indexer[0] if indexer[0] != -1 else next(
                (i for i, d in enumerate(self.net_values_series.index) if d >= start_date), None)
            filtered = self.net_values_series.iloc[start_idx: i]
        elif windows is not None:
            if windows == 0:
                filtered = self.net_values_series.iloc[0: i]
            else:
                start_index = max(0, i - windows)
                filtered = self.net_values_series.iloc[start_index: i]
        else:
            raise ValueError("Must specify either period_type or window_days")

        # 检查数据量是否满足最低要求
        key = period_type if period_type else windows
        min_required = {
            'month': 1,
            'quarter': 1,
            'year': 1,
            'current': 1,
            0: 0,
            7: 7,
            14: 14,
            30: 30,  # 对于近1月的数据，估值次数小于12次不计算
            91: 91,  # 对于近3月的数据，估值次数小于12次不计算
            182: 182,  # 对于近6月的数据，估值次数小于6次不计算
            365: 365,  # 对于近1年的数据，估值次数小于12次不计算
            730: 730,  # 对于近2年的数据，估值次数小于24次不计算
            1095: 1095  # 对于近3年的数据，估值次数小于36次不计算
        }.get(key, min_points)


        if (end_date - product_start_date).days < min_required or end_date == product_start_date:
            return None
        else:
            returns = filtered
            end_returns = self.net_values_series.values[i]
            drawdown = min((end_returns - returns) / returns)
            drawdown = round(drawdown, 4)
            return drawdown

    def drawdown(self, window=None):
        """
        优化后的计算多个窗口下的回撤的方法
        :return: DataFrame：包含每个时间节点的回撤
        """
        if window is None:
            window_list = [x for x in WINDOWS]
        period_list = ['month', 'quarter', 'year']

        # 合并周期列表
        periods_to_calculate = {f'Drawdown_{win}D': win for win in window_list}
        periods_to_calculate.update({f'Drawdown_{period}D': period for period in period_list})

        Drawdown_data = {key: [] for key in ['Date'] + list(periods_to_calculate.keys())}

        for i in range(len(self.net_values_series)):
            end_date = self.net_values_series.index[i]
            Drawdown_data['Date'].append(end_date)

            for name, period in periods_to_calculate.items():
                if isinstance(period, int):  # 如果是天数窗口
                    drawdown = self._calculator_drawdown(end_date=end_date,windows=period)
                else:  # 如果是时间段
                    drawdown = self._calculator_drawdown(end_date=end_date,period_type=period)

                Drawdown_data[name].append(drawdown)

        drawdown_df = pd.DataFrame(Drawdown_data)
        return drawdown_df

    def test(self):
        for i in range(len(self.net_values_series)):
            end_date = self.net_values_series.index[i]
            a = self._calculate_volatility(end_date=end_date, windows=0)



