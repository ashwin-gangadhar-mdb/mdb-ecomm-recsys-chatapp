from pymongo import MongoClient
import pandas as pd
import json
import os

df = pd.read_csv("/Users/ashwin.gangadhar/projects/mdb-genai/data/styles.csv")
client = MongoClient(os.getenv("MDB_CONNECTION_STR"))
col = client['sample']['kafkatest']

for ele in df.to_dict(orient="records")[:2]:
    print(json.dumps(ele))
    # print(col.insert_one(ele)
