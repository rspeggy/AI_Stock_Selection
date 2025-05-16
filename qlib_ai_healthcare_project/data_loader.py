import os
import pandas as pd
import yfinance as yf
from datetime import datetime, timedelta
from typing import Union
import time

AI_HEALTHCARE_STOCKS = {
    "GH": "Guardant Health",
    "EXAS": "Exact Sciences",
    "ILMN": "Illumina",
    "TDOC": "Teladoc Health",
    "MDT": "Medtronic"
}

FINTECH_STOCKS = {
    "PYPL": "PayPal",
    "COIN": "Coinbase",
    "AFRM": "Affirm",
    "SOFI": "SoFi",
    "UPST": "Upstart Holdings"
}

CLEAN_ENERGY_STOCKS = {
    "TSLA": "Tesla",
    "ENPH": "Enphase Energy",
    "FSLR": "First Solar",
    "PLUG": "Plug Power",
    "NEE": "NextEra Energy"
}

CLOUD_BIGDATA_STOCKS = {
    "AMZN": "Amazon",
    "MSFT": "Microsoft",
    "GOOGL": "Alphabet",
    "SNOW": "Snowflake",
    "CRM": "Salesforce"
}

SEMICONDUCTOR_STOCKS = {
    "NVDA": "NVIDIA",
    "AMD": "Advanced Micro Devices",
    "INTC": "Intel",
    "ASML": "ASML Holding",
    "TSM": "Taiwan Semiconductor Manufacturing"
}

# Merge all dictionaries for download_stock_data use
ALL_INDUSTRY_STOCKS = {}
ALL_INDUSTRY_STOCKS.update(AI_HEALTHCARE_STOCKS)
ALL_INDUSTRY_STOCKS.update(FINTECH_STOCKS)
ALL_INDUSTRY_STOCKS.update(CLEAN_ENERGY_STOCKS)
ALL_INDUSTRY_STOCKS.update(CLOUD_BIGDATA_STOCKS)
ALL_INDUSTRY_STOCKS.update(SEMICONDUCTOR_STOCKS)

def download_stock_data(symbols=None, start_date=None, end_date=None, output_dir="data/raw", max_retries=3):
    """
    Download stock data from Yahoo Finance
    
    Args:
        symbols (Union[list, dict]): List or dictionary of stock symbols, if None then use all recommended stocks
        start_date (str): Start date, if None then use one year ago
        end_date (str): End date, if None then use current date
        output_dir (str): Output directory
        max_retries (int): Maximum number of retries
    """
    if symbols is None:
        symbols = ALL_STOCKS
    
    if start_date is None:
        start_date = (datetime.now() - timedelta(days=5*365)).strftime("%Y-%m-%d")
    
    if end_date is None:
        end_date = datetime.now().strftime("%Y-%m-%d")
    
    os.makedirs(output_dir, exist_ok=True)
    
    if isinstance(symbols, list):
        symbols = {symbol: symbol for symbol in symbols}
    
    for symbol, name in symbols.items():
        retry_count = 0
        while retry_count < max_retries:
            try:
                print(f"Downloading data for {symbol} ({name})...")
                data = yf.download(symbol, start=start_date, end=end_date, progress=False)
                
                if data.empty:
                    print(f"Warning: No data available for {symbol}")
                    break
                
                # Check and handle missing columns
                required_columns = ['Open', 'High', 'Low', 'Close', 'Volume']
                if not all(col in data.columns for col in required_columns):
                    print(f"Warning: {symbol} data is missing required columns")
                    break
                
                # If Adj Close column is missing, use Close column instead
                if 'Adj Close' not in data.columns:
                    data['Adj Close'] = data['Close']
                
                output_path = os.path.join(output_dir, f"{symbol}.csv")
                data.to_csv(output_path)
                print(f"Successfully downloaded {symbol} data to {output_path}")
                break
                
            except Exception as e:
                retry_count += 1
                if retry_count == max_retries:
                    print(f"Failed to download {symbol} data after {max_retries} retries: {str(e)}")
                else:
                    print(f"Error downloading {symbol} data, retrying ({retry_count}/{max_retries}): {str(e)}")
                    time.sleep(2)  # Wait 2 seconds before retrying

def load_stock_data(file_path):
    """
    Load stock data
    
    Args:
        file_path (str): CSV file path
    
    Returns:
        pd.DataFrame: Stock data
    """
    try:
        data = pd.read_csv(file_path, index_col=0, parse_dates=True)
        return data
    except Exception as e:
        print(f"Error loading data: {str(e)}")
        return None

def get_stock_info(symbol):
    """
    Get basic stock information
    
    Args:
        symbol (str): Stock symbol
    
    Returns:
        dict: Stock information
    """
    try:
        stock = yf.Ticker(symbol)
        info = stock.info
        return {
            "symbol": symbol,
            "name": info.get("longName", ""),
            "sector": info.get("sector", ""),
            "industry": info.get("industry", ""),
            "market_cap": info.get("marketCap", 0),
            "pe_ratio": info.get("trailingPE", 0)
        }
    except Exception as e:
        print(f"Error getting stock information: {str(e)}")
        return None

if __name__ == "__main__":
    # Example usage
    print("Starting download of AI healthcare related stock data...")
    download_stock_data()
    
    # Get stock information
    for symbol in AI_HEALTHCARE_STOCKS.keys():
        info = get_stock_info(symbol)
        if info:
            print(f"\n{symbol} Information:")
            for key, value in info.items():
                print(f"{key}: {value}") 