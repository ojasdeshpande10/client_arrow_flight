import pyspark
from pyspark import StorageLevel
from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.sql.types import StringType
from pyspark.sql.window import Window
import pyarrow
import pyarrow.flight as flight
import pyarrow as pa
# from pyspark.sql.functions import col, countDistinct, udf, year, weekofyear, concat, row_number, rand, count, max

import math
from datetime import datetime
#import pytz
import json
from pprint import pprint
import argparse
import random
import re
import pandas as pd


def sendpyarrow(spark_df, server_address='130.245.132.100', port=47470):
    
    data = spark_df.collect()
    pandas_df = pd.DataFrame(data, columns=spark_df.columns)    
    client = flight.FlightClient(f'grpc://{server_address}:{port}')
    
    table = pa.Table.from_pandas(pandas_df)
    # descriptor will act as the ID for the data stream being sent
    descriptor = flight.FlightDescriptor.for_path("example_path")
    
    writer, _ = client.do_put(descriptor, table.schema)
    writer.write_table(table)
    writer.close()

    

def main():
    spark = SparkSession.builder \
        .config("spark.sql.execution.arrow.enabled", "false") \
        .config("spark.sql.execution.arrow.pyspark.enabled", "false") \
        .appName("ctlb_sampler_arrow") \
        .getOrCreate()

    spark_df = spark.read.format("csv") \
            .option("header", "true") \
            .option("inferSchema", "true") \
            .load("/user/hchoudhary/part-00199-84513dd2-ea0c-43c5-9f13-543f9ca783f9-c000.csv")
    

    spark_df.show()
    print("the number of rows in dataframe: ",spark_df.count())
    sendpyarrow(spark_df)
if __name__ == "__main__":
    main()
