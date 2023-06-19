import os
import pandas as pd
import yfinance as yf
from datetime import datetime, timedelta
import logging
import argparse
import json
import sys
import dask.dataframe as dd
import talib

STOCK_DATE_FORMAT = '%Y-%m-%d'

def setup_logging(log_folder):
    logging.basicConfig(level=logging.DEBUG,
                        format='%(asctime)s - %(levelname)s - %(message)s')

    if log_folder:
        current_date = datetime.now().strftime(STOCK_DATE_FORMAT)
        log_file_path = os.path.join(log_folder, f'download_stock_data_{current_date}.log')
        file_handler = logging.FileHandler(log_file_path)
        file_handler.setLevel(logging.DEBUG)
        file_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
        logging.getLogger().addHandler(file_handler)


def download_stock_data(stock_list_path, data_folder, log_folder=None):
    stock_list_csv = pd.read_csv(stock_list_path, header=None, names=['symbol'], dtype=str)
    stock_list = stock_list_csv['symbol'].astype(str).tolist()
    logging.info(stock_list)
    if not os.path.exists(data_folder):
        os.makedirs(data_folder)

    setup_logging(log_folder)

    for symbol in stock_list:
        file_path = os.path.join(data_folder, f"{symbol}.csv")
        stock_data_exists = os.path.isfile(file_path)
        if stock_data_exists:
            stock_data = pd.read_csv(file_path, index_col='Date', parse_dates=True)
            last_date = stock_data.index[-1].date() + timedelta(days=1) if len(stock_data) > 0 else None
        else:
            last_date = None

        start_date, end_date = get_date_range_to_download(last_date)

        if start_date is not None:
            download_and_save_stock_data(symbol, start_date, end_date, file_path)
        else:
            logging.info(f"{symbol}: data up to date, no need to download")
            return 

        stock_data = pd.read_csv(file_path, index_col='Date', parse_dates=True)
        stock_data = clean_mulip_date(stock_data)
        if stock_data is not None:
            stock_data.to_csv(file_path, mode='w', header=not os.path.isfile(file_path))


        


def get_date_range_to_download(last_date):
    if last_date is None:
        logging.info("Downloading data from 2010 to now") 
        start_date = '2010-01-01'
        end_date = datetime.now().strftime(STOCK_DATE_FORMAT)
    else:
        delta = (datetime.now().date() + timedelta(days=1) - last_date).days
        if delta > 0:
            logging.info(f"Updating data from {last_date} to {datetime.now().date()}")
            start_date = last_date.strftime(STOCK_DATE_FORMAT)
            end_date = (datetime.now() + timedelta(days=1) ).strftime(STOCK_DATE_FORMAT)
        else:
            start_date = None
            end_date = None

    return start_date, end_date


def download_and_save_stock_data(symbol, start_date, end_date, file_path):
    logging.info(f"{symbol}: Downloading data from {start_date} to {end_date}")
    # stock = yf.Ticker(symbol)
    stock_data = yf.download(f'{symbol}', start=start_date, end=end_date)
    if stock_data.empty:
        logging.warning(f"{symbol}: No data found for the given date range")
    else:
        stock_data.to_csv(file_path, mode='a', header=not os.path.isfile(file_path))
        logging.info(f"{symbol}: Data downloaded and saved successfully")


def calculate_psy(data, n=12):
    """計算 PSY 指標"""
    close = data['Close']
    psy = (close - close.shift(n)) / close.shift(n) * 100
    psy.fillna(0, inplace=True)
    return psy


def handle_date_column(stock_data):
    stock_data = stock_data.copy()
    formatted_dates = []
    for date in stock_data['Date']:
        if isinstance(date, pd.Timestamp) or isinstance(date, pd.DatetimeIndex) or isinstance(date, datetime):
            formatted_date = date.strftime(STOCK_DATE_FORMAT)
        else:
            try:
                formatted_date = datetime.strptime(date, STOCK_DATE_FORMAT).strftime(STOCK_DATE_FORMAT)
            except ValueError:
                formatted_date = datetime.strptime(date[:10], STOCK_DATE_FORMAT).strftime(STOCK_DATE_FORMAT)
        formatted_dates.append(formatted_date)

    stock_data['Date'] = formatted_dates
    return stock_data


def save_to_parquet(stock_id, stock_data, output_dir):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    stock_data = build_new_clear(stock_data)
    output_file = f'{output_dir}/{stock_id}.parquet'
    stock_data = clean_mulip_date(stock_data)
    if stock_data is not None:
        stock_data.to_parquet(output_file, engine='pyarrow', compression='snappy', index=False)
    
def build_new_clear(data):
    data= data[['Date','Open', 'High', 'Low', 'Close', 'Volume']]
    data = handle_date_column(data) 
    data['ma5'] = data['Close'].rolling(window=5).mean()
    data['ma10'] = data['Close'].rolling(window=10).mean()
    data['ma20'] = data['Close'].rolling(window=20).mean()
    data['ma60'] = data['Close'].rolling(window=60).mean()
    data['ma120'] = data['Close'].rolling(window=120).mean()
    
    # 隨機指標 K 線和 D 線
    data['STOCHk'], data['STOCHd'] = talib.STOCH(data['High'], data['Low'], data['Close'])

    # 布林帶百分比指標
    upper, middle, lower = talib.BBANDS(data['Close'], timeperiod=5, nbdevup=2, nbdevdn=2, matype=0)
    data['BBP'] = (data['Close'] - lower) / (upper - lower)

    # 加權移動平均線
    data['FWMA'] = talib.WMA(data['Close'], timeperiod=10)

    # 威廉指標
    data['WILLR'] = talib.WILLR(data['High'], data['Low'], data['Close'], timeperiod=14)

    # 乖離率短期線與長期線
    data['ISA'] = data['Close'] - talib.EMA(data['Close'], timeperiod=9)
    data['ISB'] = data['Close'] - talib.EMA(data['Close'], timeperiod=26)

    # 交叉趨勢短期線、長期線及信號線
    # 交叉趨勢信號線
    data['ITS'] = talib.EMA(data['Close'], timeperiod=9)
    data['IKS'] = talib.EMA(data['Close'], timeperiod=26)
    
    # 能量潮指標
    data['OBV'] = talib.OBV(data['Close'], data['Volume'])
    # 計算DMI指標
    data['ADX'] = talib.ADX(data['High'], data['Low'], data['Close'], timeperiod=14)

    # 計算RSI指標
    data['RSI'] = talib.RSI(data['Close'].values, timeperiod=14)
    
    # 計算PSY指標
    data['PSY'] = calculate_psy(data,12)

    data.dropna(inplace=True)
    data.reset_index(inplace=True,drop=True)
    return data
    
        
def read_csv_and_convert_to_parquet(input_dir='XXX', output_dir='output_parquet'):
    logging.info('---------------------------------------------------------------')
    logging.info('read_csv_and_convert_to_parquet')
    logging.info('---------------------------------------------------------------')
    start_date = '2010-01-01'
    end_date = '2023-12-31'
    for file in os.listdir(input_dir):
        if file.endswith('.csv'):
             
            stock_id = os.path.splitext(file)[0]
            file_path = os.path.join(input_dir, file)
            stock_data = pd.read_csv(file_path)
            stock_data = handle_date_column(stock_data)
            stock_data['Date'] = pd.to_datetime(stock_data['Date'])
            
            # Filter data between 2010 and 2023
            stock_data = stock_data[(stock_data['Date'] >= start_date) & (stock_data['Date'] <= end_date)]
            
            # Check if the number of data points matches the expected range
            date_range = pd.date_range(start_date, end_date, freq='B')
            min_data_points = int(0.6 * len(date_range))
            
            if len(stock_data) >= min_data_points:
                save_to_parquet(stock_id, stock_data, output_dir)
            else:
                logging.info(f"Data for stock {stock_id} does not meet the minimum required data points.")    

def clean_mulip_date( df ):
    df.sort_values(by='Date', inplace=True)
    # 重設索引，將 "Date" 變成一個列（column）
    df.reset_index(inplace=True)
    # 刪除重複的'Date'，只保留最後一筆
    df.drop_duplicates(subset='Date', keep='last', inplace=True)



if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Download stock price data from Yahoo Finance.')
    parser.add_argument('-c', '--config', required=True, help='path to the configuration file')
    args = parser.parse_args()
    # 讀取設定檔案
    with open(args.config) as f:
        config = json.load(f)
        stock_list_path = config['stock_list_path']
        original_data_folder = config['original_data_folder']
        log_folder = config['log_folder']
        clean_data_folder = config['clean_data_folder']
    # 下載股價資料
    download_stock_data(stock_list_path, original_data_folder, log_folder)
    read_csv_and_convert_to_parquet(input_dir =original_data_folder, output_dir= clean_data_folder)

    # python download_stock.py --config config.json