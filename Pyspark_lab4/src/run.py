import pandas as pd

from src.preprocess import read_data_csv, Preprocess_Data
from src.models import Train
from src.connect_to_db import read_data
from config import DATABASE,SERVER,PWD,UID

table_name = 'Kmeans_lab3_data' #с какой таблицы отбирать данные

data = read_data(table_name, DATABASE,SERVER, PWD, UID)
data.drop(['index'], axis=1, inplace=True)
data.to_csv('../data/temp.csv', sep="\t")
print('data saved')
df = read_data_csv('../data/temp.csv')
print("data temp readed")
vec_assembler, scaler, column_init = Preprocess_Data(df).make_data()
Train(df, vec_assembler,scaler, column_init).train_models()