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


def clear_old_connections():
    """
    清理舊的 MongoDB 連線並釋放資源。
    """
    global _client
    if _client is not None:
        try:
            # 🔍 查找當前的 active connections
            admin_db = _client.admin
            res = admin_db.command("connPoolStats")

            # 🔄 列出所有連線
            active_connections = res.get("hosts", {})
            print("🔄 發現的舊連線數量:", len(active_connections))

            # 🔥 遍歷並刪除所有舊連線
            for host, details in active_connections.items():
                print(f"🔥 刪除連線: {host}")
                admin_db.command("dropConnections", host=host)

            # 最後清理當前連接
            close_mongo_client()
            print("✅ 所有舊連線已清理完成")

        except Exception as e:
            print(f"❌ 清理舊連線時發生錯誤: {e}")
    else:
        print("⚠️ 沒有找到活躍的 MongoDB 連接")
