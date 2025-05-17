from pymongo.mongo_client import MongoClient 
import pandas as pd
import json

#url
uri = "mongodb+srv://PrathameshBhurke:Prathameshbhurke666@cluster0.ozajm.mongodb.net/?retryWrites=true&w=majority&appName=Cluster0"

# Create a new client and connect to the server
client = MongoClient(uri) 

#Create database name and collection
database_name = "pwskills"
collection_name = "waferfault"

df = pd.read_csv(r"C:\Users\Hp\Downloads\Gen AI & DS (PW Skills)\Module 36 (ML Sensor Project)\Sensor Project 01\notebooks\wafer_23012020_041211.csv")

json_records = list(json.loads(df.T.to_json()).values())

client[database_name][collection_name].insert_many(json_records)