import pyspark
from pyspark import StorageLevel
from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.sql.types import StringType
from pyspark.sql.window import Window
import pyarrow
import pyarrow.flight as flight
import pyarrow as pa
import pyarrow.fs as pafs
import time
import os
import argparse
import random
import re
import pandas as pd
import pyarrow.compute as pc
import pyarrow.parquet as pq

class MyFlightClient(flight.FlightClient):
    
    def __init__(self, server_address='130.245.132.100', port=5111):
        super().__init__(f'grpc://{server_address}:{port}')
    def exchangepyarrow(self, spark_df):
        #data = spark_df.collect()
        data = [
        {"column1": "value1", "column2": "value2"},
        {"column1": "value3", "column2": "value4"},
    ]
        pandas_df = pd.DataFrame(data)    
        table = pa.Table.from_pandas(pandas_df)
        # descriptor will act as the ID for the data stream being sent
        descriptor = flight.FlightDescriptor.for_path("example_path")
        # writer, _ = client.do_put(descriptor, table.schema)
        # writing code for exchanging the tweets and embeddings
        writer, reader  = self.do_exchange(descriptor)
        #  writing to the server
        writer.begin(table.schema)
        writer.write_table(table)
        writer.close()
        time.sleep(5)
        # receiving from the server
        received_table = reader.read_all()
        pq.write_table(received_table, 'example_embeddings.parquet')

    def sendpyarrow(self, input_table):

        # Approach 1: converting spark dataframe to pandas and then to pyarrow table
        # pandas_df = spark_df.toPandas()   
        # Approach 2: taking input as pyarrow table
        table = input_table
        # descriptor will act as the ID for the data stream being sent
        descriptor = flight.FlightDescriptor.for_path("example_path")
        writer, _ = self.do_put(descriptor, table.schema)
        writer.write_table(table)
        writer.close()
        
    def fetch_data_from_server(self, filesystem, filepath):
        # Create a ticket for the data you want. The content can be anything that your server understands.
        ticket = flight.Ticket('data_request_ticket')
        # Request the data
        reader = self.do_get(ticket)
        # Read the data into a PyArrow Table
        table = reader.read_all()
        tokenization_time = table.column('tokenization_time').to_pylist()[0]
        print(tokenization_time)
        embedding_time = table.column('embedding_time').to_pylist()[0]
        updated_table = table.drop(['tokenization_time'])
        updated_table = updated_table.drop(['embedding_time'])
        print("the column names are: ", updated_table.schema.names)
        start_write_hdfs_time = time.time()
        pq.write_table(table,filepath,filesystem=filesystem)
        end_write_hdfs_time =time.time()
        return end_write_hdfs_time-start_write_hdfs_time, tokenization_time, embedding_time



def batch_table(table, batch_size, group_column_index='year_datetime'):
    num_rows = table.num_rows
    if num_rows == 0:
        return
    # Initialize the batch start index and the end index
    start_index = 0
    current_group = None
    batch_row_count = 0
    # Iterate over each row in the table
    for i in range(num_rows):
        # Fetch the group value of the current row
        row_group = table.column(group_column_index)[i].as_py()
        # Check if we are still in the same group
        if row_group != current_group:
            # If this is not the first group and we have reached the batch size limit, yield the batch
            if current_group is not None and ((i - start_index) > batch_size):
                yield table.slice(start_index, i - start_index)
                start_index = i
                batch_row_count = 0
            current_group = row_group
        batch_row_count += 1
        # Check if it's the last row; ensure to yield the last batch
        if i == num_rows - 1:
            yield table.slice(start_index, num_rows - start_index)

def main():
    ### Reading parquet files from HDFS into py arrow tables.(without spark)
    start_time = time.time()
    os.environ['ARROW_LIBHDFS_DIR'] = '/home/hlab-admin/hadoop/lib/native/libhdfs.so'
    os.environ['CLASSPATH'] = os.popen('hadoop classpath --glob').read().strip()
    hdfs = pafs.HadoopFileSystem(host='hdfs://apollo-d0',port=9000)
    try:
        read_time_start = time.time()
        dataset = pq.ParquetDataset('/user/large-scale-embeddings/splits_for_ctlb/filter2019_0_41_50',filesystem=hdfs)
        table = dataset.read()
        read_time_end = time.time()
        column_names = table.schema.names
        print("the column names are: ", column_names)
        print("Number of rows: ", table.num_rows)
        print("Number of columns: ", table.num_columns)
    except Exception as e:
        print("error in reading the data: ",e)
    # limited_table = table.slice(0, 1000000)
    start_time_sort = time.time()
    limited_table = table.sort_by("year_datetime")
    end_time_sort = time.time()
    i=0
    network_time = 0
    write_time = 0
    total_embedding_time = 0
    total_tokenization_time = 0
    myclient = MyFlightClient()
    total_unique_groups = 0
    total_messages = 0
    for batch in batch_table(limited_table, 10000):
        unique_groups = pc.unique(batch['year_datetime'])
        total_unique_groups += len(unique_groups.to_pylist())
        total_messages += batch.num_rows
        myclient.sendpyarrow(batch)
        i+=1
        filepath = '/user/large-scale-embeddings/demo_embeddings/'+'batch'+str(i)+'_embedding_demo_test_run4_10_May_usr_yr_week_100000_rows.parquet'
        start_get_time = time.time()
        write_time_batch, tokenization_time, embedding_time = myclient.fetch_data_from_server(hdfs, filepath)
        end_get_time = time.time()
        network_time_batch = (end_get_time-start_get_time)-write_time_batch
        write_time += write_time_batch 
        network_time += network_time_batch
        total_embedding_time += embedding_time
        total_tokenization_time += tokenization_time
        if total_unique_groups > 10000:
            print("Total Unique groups for batch"+str(i)+":", total_unique_groups)
            print("Total Messages for batch"+str(i)+":", total_messages)
            break
    print("Read Time : ", read_time_end-read_time_start)
    print("Network Time : ",network_time)
    print("Write Time : ",write_time)
    print("Total Embedding Time : ",total_embedding_time)
    print("Total tokenization Time : ", total_tokenization_time)
    print("Total sort time : ", end_time_sort-start_time_sort)
    end_time = time.time()
    print("total time taken: ", end_time - start_time)

    

if __name__ == "__main__":
    main()
