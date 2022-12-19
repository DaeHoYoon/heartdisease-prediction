#%%
import pandas as pd
import pymysql
from sqlalchemy import create_engine


# db 저장 (csv>db)
def uloaddb(filepath, id, pw, host, dbname, tbname):
    df = pd.read_csv(filepath)
    dbpath = f'mysql+pymysql://{id}:{pw}@{host}/{dbname}'
    dbconn = create_engine(dbpath) # 접속
    conn = dbconn.connect()
    df.to_sql(name = tbname, con = conn, if_exists= 'fail', index=False)

# 데이터 불러오기 (db>csv)
def dloaddb(host, id, pw, dbname, tbname):
    conn = pymysql.connect(host = host, user = id, passwd= str(pw), database= dbname, charset= 'utf8')
    cur = conn.cursor()
    df = pd.read_sql(f"select * from {tbname}", con = conn)

    return df