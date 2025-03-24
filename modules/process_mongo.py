import pymongo
from modules.config_loader import load_config
from modules.logger import setup_logger


config = load_config()
logger = setup_logger()

_client = None


def get_mongo_client():
    global _client
    if _client is None:
        mongo_uri = config.get("mongo_uri", "mongodb://localhost:27017")
        _client = pymongo.MongoClient(mongo_uri, serverSelectionTimeoutMS=100000)
    return _client[config.get("mongo_db", "stock_db")]


def close_mongo_client():
    global _client
    if _client is not None:
        _client.close()
        _client = None
