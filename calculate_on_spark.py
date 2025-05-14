# _*_ coding: utf-8 _*_
"""
@Time : 2025/5/14 18:33
@Auth : Derek
@File : calculate_on_spark.py
@IDE  : PyCharm
"""

from pyspark.sql import SparkSession
from pyspark.sql import *
import os

from pyspark.sql.types import StructField, StringType, FloatType

from config import WINDOWS
from calculator import ReturnRiskIndexCalculator
import pandas as pd

os.environ['JAVA_HOME'] = "/opt/java"
os.environ['SPARK_HOME'] = "/opt/spark"

windows = WINDOWS

def process_product_partition_with_calculator(partition):
    data = list(partition)
    if not data:
        return []
    
    metrics_to_calculate = [
        'annualized_return',
        'valuation_counts'
    ]
    
    grouped = {}
    for row in data:
        prod_reg_code = row[0]
        date = row[1]
        net_val = row[2]
        if prod_reg_code not in grouped:
            grouped[prod_reg_code] = []
        grouped[prod_reg_code].append((date, net_val))
    
    output = []
    for prod_reg_code, items in grouped.items():
        dates = [item[0] for item in items]
        net_vals = [item[1] for item in items]
        series = pd.Series(net_vals, index=pd.to_datetime(dates))
        calculator = ReturnRiskIndexCalculator(series)
        result = calculator.annualized_return()
        if not result.empty:
            for _, row in result.iterrows():
                row_data = (prod_reg_code, *row.tolist())
                output.append(row_data)
        else:
            row_data = (prod_reg_code, *([None] * result.shape[1]))
            output.append(row_data)
    
    return output

def create_spark_session(input_df):
    spark_session = (SparkSession.builder
                     .appName('metrics_calculate')
                     .config("spark.hadoop.hive.metastore.uris", "thrift://192.168.1.1:21088")
                     .config("spark.driver.extraClassPath", "/opt/hadoop/postgresql.jar")
                     .config("spark.sql.warehouse.dir", "/mnt/warehouse/spark")
                     .config("hive.exec.scratchdir", "/mnt/warehouse/spark")
                     .config("spark.driver.extraJavaOptions", "-Dfile.encoding=UTF-8")
                     .config("spark.executor.extraJavaOptions", "-Dfile.encoding=UTF-8")
                     .config("spark.sql.shuffle.partitions", "5000")
                     .config("spark.default.parallelism", "2000")
                     .config("spark.driver.maxResultSize", "4096m")
                     .config("spark.sql.debug.maxToStringFields", "1000")
                     .enableHiveSupport()
                     .getOrCreate())
    print('spark session created')
    repartitioned_df = spark_session.createDataFrame(input_df)
    print('repartitioned dataframe created')
    repartitioned_df = repartitioned_df.repartition("prod_reg_code")
    repartitioned_df.show()
    print('repartitioned dataframe show')
    
    fields = [
        StructField("prod_reg_code", StringType(), True),
        StructField("net_val_date", StringType(), True),
        StructField("d7_returns", FloatType(), True),
        StructField("d14_returns", FloatType(), True),
        StructField("m1_returns", FloatType(), True),
        StructField("m3_returns", FloatType(), True),
        StructField("m6_returns", FloatType(), True),
        StructField("y1_returns", FloatType(), True),
    ]
    schema = StringType(fields)
    print('repartitioned dataframe schema')
    
    final_df = repartitioned_df.rdd.mapPartitions(process_product_partition_with_calculator).toDF(schema)
    print('final dataframe created')
    
    jdbc_url = "kdbc:postgresql://192.168.1.1:5432/postgres"
    table = "bdm.returns"
    username = "monitor"
    password = "monitor"
    driver = "org.postgresql.Driver"
    
    (final_df.write.format("jdbc")
     .option("url", jdbc_url)
     .option("dbtable", table)
     .option("user", username)
     .option("password", password)
     .option("driver", driver)
     .option("numPartitions", 100)
     .option("batchsize", 10000)
     .mode("overwrite")
     .save())
    
    spark_session.stop()
