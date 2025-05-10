from data_parser import parse_input_to_series
from calculator import ReturnRiskIndexCalculator
from config import INPUT_STR


def main():
    series = parse_input_to_series(INPUT_STR)
    calculator = ReturnRiskIndexCalculator(series)
    #
    # # 计算年化收益率
    # annualized_returns_df = calculator.annualized_return()
    # print("Annualized Returns:\n", annualized_returns_df)
    #
    # # 估值次数计算
    # valuation_df, valuation_output = calculator.count_valuation()
    # print("Valuation Count DataFrame:\n", valuation_df)
    #
    # 年化波动率计算
    # volatility = calculator.annualized_volatility()
    # print("Volatility Returns:\n", volatility)
    #
    # 夏普比率计算
    # sharpe_ratio = calculator.annualized_sharpe_ratio()
    # print("Sharpe Ratio:\n", sharpe_ratio)

    # 最大回撤计算
    # max_drawdown = calculator.max_drawdown()
    # print("Max Drawdown:\n", max_drawdown[['Date', 'Max_drawdown_7D']])

    # 卡玛比率
    calmer_ratio = calculator.annualized_calmer_ratio()
    print("Calmer Ratio:", calmer_ratio)

    # # test
    # test = calculator.test()
    # print(test)


if __name__ == "__main__":
    main()