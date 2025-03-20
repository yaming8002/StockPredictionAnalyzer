import pymongo
from modules.config_loader import load_config
from modules.logger import setup_logger


config = load_config()
logger = setup_logger()


def get_mongo_client():
    """從 config.json 取得 MongoDB 設定"""
    mongo_uri = config.get("mongo_uri", "mongodb://localhost:27017")
    client = pymongo.MongoClient(mongo_uri)
    return client[config.get("mongo_db", "stock_db")]
