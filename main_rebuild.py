import pandas as pd
from calculator import ReturnRiskIndexCalculator
from input_df import create_df

def create_calculator_from_dataframe(df, prod_reg_code):
    """ 根据给定的产品代码从DataFrame中抽取数据，并创建ReturnRiskIndexCalculator实例 """
    product_data = df[df['prod_reg_code'] == prod_reg_code]
    series = pd.Series(product_data['net_val'].values,
                       index=pd.to_datetime(product_data['date']))
    return ReturnRiskIndexCalculator(series)

def calculate_metrics_for_product(df, prod_reg_code):
    calculator = create_calculator_from_dataframe(df, prod_reg_code)

    # 分别调用两个指标方法
    all_metrics = calculator.calculate_all_metrics()
    # 初始化一个空的DataFrame用于合并所有指标
    combined_df = None
    for metric_name, metric_df in all_metrics.items():
        if combined_df is None:
            combined_df = metric_df
        else:
            combined_df = pd.merge(combined_df, metric_df, on='Date', how='outer')

    # 添加产品代码并调整列顺序，使 prod_reg_code 成为第二列
    combined_df['prod_reg_code'] = prod_reg_code
    cols = combined_df.columns.tolist()
    date_idx = cols.index('Date')
    cols = cols[:date_idx + 1] + ['prod_reg_code'] + cols[date_idx + 1:-1]
    combined_df = combined_df[cols]

    return combined_df

def main(input_df):
    results = []
    for prod_code in input_df['prod_reg_code'].unique():
        result = calculate_metrics_for_product(input_df, prod_code)
        results.append(result)

    if results:
        final_df = pd.concat(results, ignore_index=True)
        print(final_df)
        return final_df
    else:
        return pd.DataFrame(columns=['prod_reg_code', 'original_date',
                                     'Volatility_0D', 'ValuationFrequency_0D'])

if __name__ == '__main__':
    input_str = ("20250301^6,20250302^6,20250303^2,20250304^2,20250305^8,20250306^5,"
                 "20250307^8,20250308^3,20250309^1,20250310^3,20250311^3,20250312^9,"
                 "20250313^7,20250314^4,20250315^1,20250316^1,20250317^7,20250318^3,"
                 "20250319^7,20250320^3,20250321^8,20250322^7,20250323^1,20250324^3,"
                 "20250325^2,20250326^8,20250327^9,20250328^1,20250329^1,20250330^3,"
                 "20250331^7,20250401^6")
    input_df = create_df(input_str)
    main(input_df)