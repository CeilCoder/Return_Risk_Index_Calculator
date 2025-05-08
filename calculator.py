import pandas as pd
from utils import calculate_interval_return, annualized_return, get_quarter_days
from config import WINDOWS
from datetime import timedelta


class ReturnRiskIndexCalculator:
    def __init__(self, net_values_series):
        self.net_values_series = net_values_series

    def batch_calculate_returns(self):
        returns = self.net_values_series.pct_change()
        return returns.dropna()  # 删除第一个NaN值

    def annualized_return(self):
        result_data = []
        for dt in self.net_values_series.index:
            row = {"Date": dt.strftime("%Y%m%d")}
            # 当前与前一天比较
            if dt != self.net_values_series.index[0]:
                prev_dt = self.net_values_series.index[self.net_values_series.index.get_loc(dt) - 1]
                interval_return = calculate_interval_return(self.net_values_series.loc[dt],
                                                            self.net_values_series.loc[prev_dt])
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