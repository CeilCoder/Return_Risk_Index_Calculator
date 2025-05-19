from datetime import timedelta

import numpy as np
from config import WINDOWS, INPUT_STR
from datetime import datetime, timedelta, date

class ReturnRiskCalculator:
    def __init__(self, net_values_series):
        self.net_values_series = net_values_series
        sorted_items = sorted(
            net_values_series.items(),
            key=lambda x: datetime.strptime(x[0], "%Y%m%d").date()
        )
        self.date_objs = [datetime.strptime(d, "%Y%m%d").date() for d, _ in sorted_items]
        self.value_list = [v for _, v in sorted_items]

    # ----------- 起始日期计算相关方法 --------------

    def _get_start_date(self, end_date, period_type=None, windows=None, index_list=None):
        if period_type:
            if isinstance(end_date, str):
                end_date = datetime.strptime(end_date, "%Y%m%d").date()
            if period_type == 'month':
                first_day_of_month = datetime(end_date.year, end_date.month, 1)
                start_date = (first_day_of_month - timedelta(days=1)).date()
            elif period_type == 'quarter':
                q_start_month = ((end_date.month - 1) // 3) * 3 + 1
                first_day_of_quarter = datetime(end_date.year, q_start_month, 1)
                start_date = (first_day_of_quarter - timedelta(days=1)).date()
            elif period_type == 'year':
                first_day_of_year = datetime(end_date.year, 1, 1)
                start_date = (first_day_of_year - timedelta(days=1)).date()
            elif period_type == 'current':
                start_date = end_date - timedelta(days=1)
            else:
                raise ValueError("Unsupported period_type")
            return start_date
        elif windows is not None:
            if index_list is None:
                raise ValueError("Must provide index_list")
            if isinstance(end_date, str):
                end_date = datetime.strptime(end_date, "%Y%m%d").date()
            for i, d in enumerate(index_list):
                if isinstance(d, str):
                    d = datetime.strptime(d, "%Y%m%d").date()
                if d == end_date:
                    loc = i
                    break
            else:
                raise ValueError("Must provide index_list")
            start_index = max(0, loc - windows)
            start_date = index_list[start_index]

            if isinstance(start_date, str):
                start_date = datetime.strptime(start_date, "%Y%m%d").date()

            return start_date
        else:
            raise ValueError("Must provide period_type or windows day")

    def _filter_data_by_dates(self, start_date, end_date):
        if isinstance(start_date, str):
            start_date = datetime.strptime(start_date, "%Y%m%d").date()
        if isinstance(end_date, str):
            end_date = datetime.strptime(end_date, "%Y%m%d").date()

        sorted_items = sorted(self.net_values_series.items(), key=lambda x: datetime.strptime(x[0], '%Y%m%d').date())

        filtered_data = []
        for d, v in sorted_items:
            current_date = datetime.strptime(d, '%Y%m%d').date()
            if start_date <= current_date <= end_date:
                filtered_data.append((current_date, v))

        return filtered_data

    # ----------- 单项指标计算 --------------

    # ----------- 年化收益率计算相关方法 --------------

    def _calculate_annualized_returns(self, end_date, period_type=None, windows=None, min_points=None):
        product_start_date = self.date_objs[0]

        if windows == 0:
            start_date = product_start_date
        else:
            start_date = self._get_start_date(end_date, period_type, windows, index_list=self.date_objs)

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

        start_date, start_value = filtered[0]
        end_date, end_value = filtered[-1]
        days_diff = (end_date - start_date).days

        if len(filtered) < min_required or days_diff < min_required:
            return None
        else:
            returns = (end_value - start_value) / start_value
            returns_year = returns * (365 / days_diff)

        return returns_year

    # ----------- 估值次数计算相关方法 --------------

    def _calculate_valuation_count(self, end_date, windows=None, period_type=None, min_points=None):
        # 确定起始日期
        product_start_date = self.date_objs[0]

        if windows == 0:
            start_date = product_start_date
        else:
            start_date = self._get_start_date(end_date, period_type, windows, index_list=self.date_objs)

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

        if (end_date - product_start_date).days < 0:
            return None
        else:
            return len(filtered)

    # ----------- 年化波动率计算相关方法 --------------

    def _calculate_volatility(self, end_date, period_type=None, windows=None, freq_multiplier=365, min_points=12):
        if isinstance(end_date, str):
            end_date = datetime.strptime(end_date, "%Y%m%d").date()
        else:
            end_date = end_date
        # 找到 <= end_date 的所有数据
        series_end_idx = None
        for i, d in enumerate(self.date_objs):
            if d > end_date:
                break
            series_end_idx = i
        if series_end_idx is None or series_end_idx < 1:
            return None, None
        # 确定起始日期
        product_start_date = self.date_objs[0]

        if period_type:
            if period_type == 'month':
                start_date = date(end_date.year, end_date.month, 1) - timedelta(days=1)
            elif period_type == 'quarter':
                q_start_month = ((end_date.month - 1) // 3) * 3 + 1
                start_date = date(end_date.year, q_start_month, 1) - timedelta(days=1)
            elif period_type == 'year':
                start_date = date(end_date.year, 1, 1) - timedelta(days=1)
            elif period_type == 'current':
                start_date = end_date - timedelta(days=1)
            else:
                raise ValueError(f"Unsupported period_type: {period_type}")

            # 找到第一个 >= start_date 的索引
            start_idx = None
            for i, d in enumerate(self.date_objs):
                if d >= start_date:
                    start_idx = i
                    break
            if start_idx is None or start_idx > series_end_idx:
                return None, None

            filtered_dates = self.date_objs[start_idx:series_end_idx + 1]
            filtered_values = self.value_list[start_idx:series_end_idx + 1]

        elif windows is not None:
            if windows == 0:
                start_idx = 0
            else:
                start_idx = max(0, series_end_idx - windows)
            filtered_dates = self.date_objs[start_idx:series_end_idx + 1]
            filtered_values = self.value_list[start_idx:series_end_idx + 1]
        else:
            raise ValueError("Must specify either period_type or window_days")

        # 检查数据量是否满足最低要求
        key = period_type if period_type else windows
        min_required = {
            'month': 12,
            'quarter': 12,
            'year': 12,
            0: 182,
            30: 12,  # 对于近1月的数据，估值次数小于12次不计算
            91: 12,  # 对于近3月的数据，估值次数小于12次不计算
            182: 6,  # 对于近6月的数据，估值次数小于6次不计算
            365: 12,  # 对于近1年的数据，估值次数小于12次不计算
            730: 24,  # 对于近2年的数据，估值次数小于24次不计算
            1095: 36  # 对于近3年的数据，估值次数小于36次不计算
        }.get(key, min_points)

        if len(filtered_values) < min_required:
            return None, None

        returns = []
        date_diffs = []
        for i in range(1, len(filtered_values)):
            prev_value = filtered_values[i - 1]
            curr_value = filtered_values[i]
            prev_date = filtered_dates[i - 1]
            curr_date = filtered_dates[i]
            if prev_value == 0:
                continue
            pct_change = (curr_value - prev_value) / prev_value
            days_diff = (curr_date - prev_date).days
            if days_diff == 0:
                continue
            returns.append(pct_change)
            date_diffs.append(days_diff)

        if not returns:
            return None, None

        r_t_period = [ret / diff for ret, diff in zip(returns, date_diffs)]
        r_period_bar = sum(r_t_period) / len(r_t_period)

        # 条件映射表
        conditions = {
            'month': [(12, freq_multiplier)],
            'quarter': [(36, freq_multiplier), (12, 52)],
            'year': [(144, freq_multiplier), (48, 52), (12, 12)],
            0: [(72 * int(len(filtered_values) / 182), freq_multiplier), (24 * int(len(filtered_values) / 182), 52),
                (6 * int(len(filtered_values) / 182), 12)],
            91: [(36, freq_multiplier), (12, 52)],
            182: [(72, freq_multiplier), (24, 52), (6, 12)],
            365: [(144, freq_multiplier), (48, 52), (12, 12)],
            730: [(288, freq_multiplier), (96, 52), (24, 12)],
            1095: [(432, freq_multiplier), (144, 52), (36, 12)]
        }

        divisors = {
            freq_multiplier: 365,
            52: 7,
            12: 30
        }

        freq = freq_multiplier
        if key in conditions:
            for threshold, frequency in conditions[key]:
                if len(filtered_values) >= threshold:
                    r_t_period = [ret / (diff / divisors[freq]) for ret, diff in zip(returns, date_diffs)]
                    freq = frequency

        # 调整收益率
        r_period_bar = np.mean(r_t_period)

        # 计算标准差和年化波动率
        variance = np.var(r_t_period, ddof=1)
        volatility = float(np.sqrt(variance) * np.sqrt(freq))
        return volatility

    # ----------- 指标聚合 --------------

    def calculate_annualized_return(self, windows=None):
        """
        计算多个窗口下的年化收益率，并返回包含所有结果的DataFrame
        参数:
        - window_days_list: list of int, 多个窗口天数，如 [30, 91]
        返回:
        - {[date:(period,result),(period,result)]}, 每列为对应窗口的收益率
        """
        if windows is None:
            window_days_list = [x for x in WINDOWS]
        period_list = ['month', 'quarter', 'year', 'current']
        # 构造 period_name -> period_value 映射
        periods_to_calculate = {}
        for win in window_days_list:
            periods_to_calculate[f"Returns_{win}D"] = win
        for period in period_list:
            periods_to_calculate[f"Returns_{period}D"] = period
        # 结果存储
        result_dict = {}

        for i, date_str in enumerate(self.date_objs):
            end_date = self.date_objs[i]
            result_list = []

            for name, period in periods_to_calculate.items():
                if isinstance(period, int):  # 天数窗口
                    returns = self._calculate_annualized_returns(end_date=end_date, windows=period)
                else:  # 时间段类型
                    returns = self._calculate_annualized_returns(end_date=end_date, period_type=period)

                result_list.append((name, returns))

            result_dict[date_str.strftime('%Y%m%d')] = result_list

        return result_dict