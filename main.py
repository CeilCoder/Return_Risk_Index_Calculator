from data_parser import parse_input_to_series
from calculator import ReturnRiskIndexCalculator
from config import INPUT_STR
import pandas as pd


def main():
    series = parse_input_to_series(INPUT_STR)
    calculator = ReturnRiskIndexCalculator(series)

    # 计算区间收益率
    interval_returns = calculator.batch_calculate_returns()
    print("Interval Returns:\n", interval_returns)

    # 计算年化收益率
    annualized_returns_df = calculator.annualized_return()
    print("Annualized Returns:\n", annualized_returns_df)

    # 估值次数计算
    valuation_df, valuation_output = calculator.count_valuation()
    print("Valuation Count DataFrame:\n", valuation_df)

    # 年化波动率计算
    volatility = calculator.combined_volatility()
    print("Volatility Returns:\n", volatility)


if __name__ == "__main__":
    main()