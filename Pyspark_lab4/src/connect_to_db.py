import pandas as pd
import pyodbc
from sqlalchemy import create_engine, event
from urllib.parse import quote_plus

"""функция для чтения данных из БД MS SQL SERVER"""

def read_data(table_name,DATABASE,SERVER,PWD,UID):
    conn = "DRIVER={{ODBC Driver 17 for SQL Server}};SERVER={0};DATABASE={1};UID={2};PWD={3}".format(
        SERVER, DATABASE, UID, PWD)

    connection = pyodbc.connect(conn)
    cursor = connection.cursor()
    quoted = quote_plus(conn)
    new_con = 'mssql+pyodbc:///?odbc_connect={}'.format(quoted)
    engine = create_engine(new_con)

    """помогает нам соедениться с бд по средством pandas поскольку пандас только может работать с SQL Alchemy"""
    @event.listens_for(engine, 'before_cursor_execute')
    def receive_before_cursor_execute(conn, cursor, statement, params, context, executemany):
        print("FUNC call")
        if executemany:
            cursor.fast_executemany = True
    """запрос отобрать все с заданной таблицы"""
    data = pd.read_sql(f'select * from {table_name}', engine)
    print("data readed")
    return data
