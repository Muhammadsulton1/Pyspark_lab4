from pyspark.ml import Transformer
import os
import sys
os.environ['PYSPARK_PYTHON'] = sys.executable
os.environ['PYSPARK_DRIVER_PYTHON'] = sys.executable

import pyspark
from pyspark import SparkContext, SparkConf
from pyspark.sql import SparkSession
from pyspark.sql.functions import col
from pyspark.ml.feature import VectorAssembler, MinMaxScaler



conf = SparkConf()
conf.set("spark.ui.port", "4050")
conf.set("spark.app.name", "third_lab_danilov")
conf.set("spark.master", "local")
conf.set("spark.executor.cores", "4")
conf.set("spark.worker.cores", "8")
conf.set("spark.executor.instances", "1")
conf.set("spark.executor.memory", "16g")
conf.set("spark.locality.wait", "0")
conf.set("spark.serializer", "org.apache.spark.serializer.KryoSerializer")
conf.set("spark.kryoserializer.buffer.max", "2000")
conf.set("spark.executor.heartbeatInterval", "6000s")
conf.set("spark.network.timeout", "10000000s")
conf.set("spark.shuffle.spill", "true")
conf.set("spark.driver.memory", "16g")
conf.set("spark.driver.maxResultSize", "16g")

# create the context
sc = SparkContext.getOrCreate(conf=conf)
spark = SparkSession.builder.getOrCreate()

"""для отбора полонок"""
class Select(Transformer):
    def __init__(self, column):
        self.column = column

    def _transform(self, data):
        if self.column is None:
            return data
        data = data.select(*self.column)
        return data

"""функция для чтения данных из файла csv"""
def read_data_csv(path):
    df = spark.read.csv(path, sep="\t", header=True, inferSchema=True)
    df = df.select([col(column).cast('float') for column in df.columns])
    df = df.na.fill(0)
    return df

"""функция для подготовки данных к векторизации, стандартизации и отбору признаков"""
class Preprocess_Data:
    def __init__(self, df):
        self.df = df
    def make_data(self):
        vec_assembler = VectorAssembler(inputCols=self.df.columns, outputCol='features')
        scaler = MinMaxScaler(inputCol='features', outputCol='standardized')
        label = 'labels'
        column_select = ['standardized', label]
        column_init = Select(column_select)
        return vec_assembler, scaler, column_init