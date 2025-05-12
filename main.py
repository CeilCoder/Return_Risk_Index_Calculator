from data_parser import parse_input_to_series
from calculator import ReturnRiskIndexCalculator
from config import INPUT_STR


def main():
    series = parse_input_to_series(INPUT_STR)
    calculator = ReturnRiskIndexCalculator(series)

    # # 计算年化收益率
    # annualized_returns_df = calculator.annualized_return()
    # print("Annualized Returns:\n", annualized_returns_df[['Date', 'Returns_7D']])

    # # 估值次数计算
    # valuation_df = calculator.valuation_count()
    # print("Valuation Count DataFrame:\n", valuation_df[['Date', 'Count_0D']])
    #
    # 年化波动率计算
    volatility = calculator.annualized_volatility()
    print("Volatility Returns:\n", volatility[['Date', 'Volatility_0D']])
    #
    # # 夏普比率计算
    # sharpe_ratio = calculator.annualized_sharpe_ratio()
    # print("Sharpe Ratio:\n", sharpe_ratio)
    #
    # # 最大回撤计算
    # max_drawdown = calculator.max_drawdown()
    # print("Max Drawdown:\n", max_drawdown[['Date', 'Max_drawdown_0D']])
    #
    # # 卡玛比率
    # calmer_ratio = calculator.annualized_calmer_ratio()
    # print("Calmer Ratio:\n", calmer_ratio[['Date', 'Calmer_0D']])
    #
    # # 回撤计算
    # drawdown = calculator.drawdown()
    # print("Drawdown:\n", drawdown[['Date', 'Drawdown_0D']])

    # test = calculator.test()


if __name__ == "__main__":
    main()