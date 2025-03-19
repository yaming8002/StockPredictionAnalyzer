def get_mongo_client():
    """從 config.json 取得 MongoDB 設定"""
    mongo_uri = config.get("mongo_uri", "mongodb://localhost:27017")
    client = pymongo.MongoClient(mongo_uri)
    return client[config.get("mongo_db", "stock_db")]
