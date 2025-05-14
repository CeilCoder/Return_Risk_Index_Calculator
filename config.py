from pyspark.sql.types import *

INPUT_STR = "20250301^6,20250302^6,20250303^2,20250304^2,20250305^8,20250306^5,20250307^8,20250308^3,20250309^1,20250310^3,20250311^3,20250312^9,20250313^7,20250314^4,20250315^1,20250316^1,20250317^7,20250318^3,20250319^7,20250320^3,20250321^8,20250322^7,20250323^1,20250324^3,20250325^2,20250326^8,20250327^9,20250328^1,20250329^1,20250330^3,20250331^7,20250401^6"

WINDOWS = [7, 14, 30, 91, 182, 365, 730, 1095, 0] # 窗口周期

SPARK_CONFIG = {
    "app_name": "metrics_calculate",
    "hive_metastore_uris": "thrift://192.168.1.1:21088",
    "extra_class_path": "/opt/hadoop/postgresql.jar",
    "warehouse_dir": "/mnt/warehouse/spark",
    "scratch_dir": "/mnt/warehouse/spark",
    "driver_java_options": "-Dfile.encoding=UTF-8",
    "executor_java_options": "-Dfile.encoding=UTF-8",
    "shuffle_partitions": 5000,
    "parallelism": 2000,
    "driver_max_result_size": "4096m",
    "debug_max_to_string_fields": 1000,
    "enable_hive_support": True,
}

# JDBC 配置
JDBC_CONFIG = {
    "url": "jdbc:postgresql://192.168.1.1:5432/postgres",
    "table": "bdm.returns",
    "user": "monitor",
    "password": "monitor",
    "driver": "org.postgresql.Driver",
    "numPartitions": 100,
    "batchsize": 10000,
    "mode": "overwrite"
}

# 输出字段 Schema 定义
OUTPUT_SCHEMA = [
    ("prod_reg_code", StringType()),
    ("net_val_date", StringType()),
    ("d7_returns", FloatType()),
    ("d14_returns", FloatType()),
    ("m1_returns", FloatType()),
    ("m3_returns", FloatType()),
    ("m6_returns", FloatType()),
    ("y1_returns", FloatType())
]