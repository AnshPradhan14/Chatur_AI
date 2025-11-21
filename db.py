import os
import certifi
from pymongo import MongoClient
from dotenv import load_dotenv

load_dotenv()

MONGO_URI = os.getenv("MONGO_URI")
DB_NAME = os.getenv("MONGO_DB_NAME")

client = MongoClient(MONGO_URI, tlsCAFile=certifi.where())
db = client[DB_NAME]

users_col = db["users"]
chats_col = db["chats"]
docs_col = db["documents"]
embeddings_col = db["embeddings"]
messages_col = db["messages"]
