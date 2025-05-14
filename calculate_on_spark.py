# _*_ coding: utf-8 _*_
"""
@Time : 2025/5/14 18:33
@Auth : Derek
@File : calculate_on_spark.py
@IDE  : PyCharm
"""

from pyspark.sql import *
import os
from pyspark.sql.types import *
from config import WINDOWS, SPARK_CONFIG, JDBC_CONFIG, OUTPUT_SCHEMA
from calculator import ReturnRiskIndexCalculator
import pandas as pd

os.environ['JAVA_HOME'] = "/opt/java"
os.environ['SPARK_HOME'] = "/opt/spark"

windows = WINDOWS

def process_product_partition_with_calculator(partition):
    data = list(partition)
    if not data:
        return []

    
    grouped = {}
    for row in data:
        prod_reg_code = row[0]
        date = row[1]
        net_val = row[2]
        if prod_reg_code not in grouped:
            grouped[prod_reg_code] = []
        grouped[prod_reg_code].append((date, net_val))
    
    output = []

    # metrics_methods = ReturnRiskIndexCalculator.get_all_metrics()

    all_combined_dfs = []

    for prod_reg_code, items in grouped.items():
        dates = [item[0] for item in items]
        net_vals = [item[1] for item in items]
        series = pd.Series(net_vals, index=pd.to_datetime(dates))

        # 使用统一接口
        calculator = ReturnRiskIndexCalculator(series)
        all_metrics = calculator.calculate_all_metrics()

        main_df = None
        added_metrics = {}

        for metric_name, result in all_metrics.items():
            if isinstance(result, pd.DataFrame):
                if main_df is None:
                    result['prod_reg_code'] = prod_reg_code
                    cols = result.columns.tolist()
                    date_idx = cols.index('Date')
                    cols = ['prod_reg_code'] + cols[:date_idx + 1] + cols[date_idx + 1:-1]
                    main_df = result[cols]
                    added_metrics[metric_name] = result.iloc[:, -1]  # 假设最后一列是指标值
                else:
                    # 合并其他 DataFrame 的指标值，不重复 Date 和 prod_reg_code
                    col_name = f"{metric_name}"
                    main_df[col_name] = result.iloc[:, -1]
            elif isinstance(result, (int, float)):
                # 标量指标：添加为新列
                col_name = f"{metric_name}"
                main_df[col_name] = result
            else:
                # 处理异常情况
                col_name = f"{metric_name}"
                main_df[col_name] = None
        if main_df is not None:
            for _, row in main_df.iterrows():
                output.append(tuple(row.values))
            #     for _, row in result.iterrows():
            #         row_data = (prod_reg_code, metric_name, *[row[col] for col in result.columns])
            #         output.append(row_data)
            # elif isinstance(result, (int, float, type(None))):
            #     row_data = (prod_reg_code, metric_name, result)
            #     output.append(row_data)
            # else:
            #     row_data = (prod_reg_code, metric_name, None)
            #     output.append(row_data)

        # metrics_results = {}
        #
        # for method_name in metrics_methods:
        #     try:
        #         func = getattr(calculator, method_name)
        #         result = func()
        #         metrics_results[method_name] = result
        #     except Exception as e:
        #         metrics_results[method_name] = None
        #
        # for method_name, result in metrics_results.items():
        #     if isinstance(result, pd.DataFrame):
        #         # 如果是DataFrame，每一行代表一个单独的结果
        #         for _, row in result.iterrows():
        #             row_data = (prod_reg_code, method_name, *[row[col] for col in result.columns])
        #             output.append(row_data)
        #     elif isinstance(result, (int, float)):
        #         # 如果是单个数值，则直接添加
        #         row_data = (prod_reg_code, method_name, result)
        #         output.append(row_data)
        #     else:
        #         # 处理其他类型或None的情况
        #         row_data = (prod_reg_code, method_name, None)
        #         output.append(row_data)
    
    return output


def build_spark_session():
    """
    构建并返回一个配置好的 SparkSession 实例。
    """
    builder = SparkSession.builder \
        .appName(SPARK_CONFIG["app_name"]) \
        .config("spark.hadoop.hive.metastore.uris", SPARK_CONFIG["hive_metastore_uris"]) \
        .config("spark.driver.extraClassPath", SPARK_CONFIG["extra_class_path"]) \
        .config("spark.sql.warehouse.dir", SPARK_CONFIG["warehouse_dir"]) \
        .config("hive.exec.scratchdir", SPARK_CONFIG["scratch_dir"]) \
        .config("spark.driver.extraJavaOptions", SPARK_CONFIG["driver_java_options"]) \
        .config("spark.executor.extraJavaOptions", SPARK_CONFIG["executor_java_options"]) \
        .config("spark.sql.shuffle.partitions", str(SPARK_CONFIG["shuffle_partitions"])) \
        .config("spark.default.parallelism", str(SPARK_CONFIG["parallelism"])) \
        .config("spark.driver.maxResultSize", SPARK_CONFIG["driver_max_result_size"]) \
        .config("spark.sql.debug.maxToStringFields", str(SPARK_CONFIG["debug_max_to_string_fields"]))

    if SPARK_CONFIG["enable_hive_support"]:
        builder = builder.enableHiveSupport()

    spark = builder.getOrCreate()
    print('Spark session created')
    return spark

def create_spark_session():
    spark = build_spark_session()

    df = spark.read \
        .format("jdbc") \
        .option("url", "jdbc:mysql://your_host:3306/your_db") \
        .option("dbtable", "your_table") \
        .option("user", "username") \
        .option("password", "password") \
        .option("driver", "com.mysql.cj.jdbc.Driver") \
        .load()

    df = df.select(
        df["prod_reg_code"].cast(StringType()),
        df["date"].cast(DateType()),
        df["net_val"].cast(DoubleType())
    ).dropna()

    repartitioned_df = spark.createDataFrame(df)
    print('repartitioned dataframe created')
    repartitioned_df = repartitioned_df.repartition("prod_reg_code")
    repartitioned_df.show()
    print('repartitioned dataframe show')
    
    schema = StructType([StructField(name, data_type, True) for name, data_type in OUTPUT_SCHEMA])
    print('repartitioned dataframe schema')
    
    final_df = repartitioned_df.rdd.mapPartitions(process_product_partition_with_calculator).toDF(schema)
    print('final dataframe created')
    
    (final_df.write.format("jdbc")
     .option("url", JDBC_CONFIG["url"])
     .option("dbtable", JDBC_CONFIG["table"])
     .option("user", JDBC_CONFIG["username"])
     .option("password", JDBC_CONFIG["password"])
     .option("driver", JDBC_CONFIG["driver"])
     .option("numPartitions", JDBC_CONFIG["numPartitions"])
     .option("batchsize", JDBC_CONFIG["batchsize"])
     .mode(JDBC_CONFIG["mode"])
     .save())
    
    spark.stop()
