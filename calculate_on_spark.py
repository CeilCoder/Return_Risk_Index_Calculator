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

    metrics_methods = ReturnRiskIndexCalculator.get_all_metrics()

    for prod_reg_code, items in grouped.items():
        dates = [item[0] for item in items]
        net_vals = [item[1] for item in items]
        series = pd.Series(net_vals, index=pd.to_datetime(dates))
        calculator = ReturnRiskIndexCalculator(series)

        metrics_results = {}

        for method_name in metrics_methods:
            try:
                func = getattr(calculator, method_name)
                result = func()
                metrics_results[method_name] = result
            except Exception as e:
                metrics_results[method_name] = None

        for method_name, result in metrics_results.items():
            if isinstance(result, pd.DataFrame):
                # 如果是DataFrame，每一行代表一个单独的结果
                for _, row in result.iterrows():
                    row_data = (prod_reg_code, method_name, *[row[col] for col in result.columns])
                    output.append(row_data)
            elif isinstance(result, (int, float)):
                # 如果是单个数值，则直接添加
                row_data = (prod_reg_code, method_name, result)
                output.append(row_data)
            else:
                # 处理其他类型或None的情况
                row_data = (prod_reg_code, method_name, None)
                output.append(row_data)
    
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

def create_spark_session(input_df):
    spark = build_spark_session()
    repartitioned_df = spark.createDataFrame(input_df)
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
