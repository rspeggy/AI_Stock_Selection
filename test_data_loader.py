import unittest
import pandas as pd
from datetime import datetime, timedelta
from data_loader import (
    download_stock_data, 
    load_stock_data, 
    get_stock_info,
    AI_HEALTHCARE_STOCKS,
    SEMICONDUCTOR_STOCKS,
    CLEAN_ENERGY_STOCKS,
    FINTECH_STOCKS,
    CLOUD_BIGDATA_STOCKS,
    ALL_INDUSTRY_STOCKS
)
from macro_data_loader import load_macro_data
import os

class TestDataLoader(unittest.TestCase):
    """Data loading module tests"""
    
    def setUp(self):
        """Test preparation"""
        # Use stocks from all industries
        self.symbols = list(ALL_INDUSTRY_STOCKS.keys())
        self.start_date = (datetime.now() - timedelta(days=3000)).strftime("%Y-%m-%d")  # 10 months
        self.end_date = datetime.now().strftime("%Y-%m-%d")
        self.output_dir = "data/raw"
    
    def test_download_all_stocks(self):
        """Test downloading data for all industry stocks"""
        # Download data for all stocks
        download_stock_data(ALL_INDUSTRY_STOCKS, self.start_date, self.end_date, self.output_dir)
        
        # Check if all stock files exist
        for symbol in ALL_INDUSTRY_STOCKS.keys():
            file_path = f"{self.output_dir}/{symbol}.csv"
            self.assertTrue(os.path.exists(file_path), f"Data file for stock {symbol} does not exist")
            
            # Check data format
            data = load_stock_data(file_path)
            self.assertIsInstance(data, pd.DataFrame, f"Data for stock {symbol} is not in DataFrame format")
            self.assertFalse(data.empty, f"Data for stock {symbol} is empty")
            self.assertTrue(all(col in data.columns for col in ['Open', 'High', 'Low', 'Close', 'Volume']),
                          f"Data for stock {symbol} is missing required columns")
    
    def test_download_stock_data(self):
        """Test stock data download"""
        # Download data
        download_stock_data(self.symbols, self.start_date, self.end_date, self.output_dir)
        
        # Check if files exist
        for symbol in self.symbols:
            file_path = f"{self.output_dir}/{symbol}.csv"
            self.assertTrue(os.path.exists(file_path))
            
            # Check data format
            data = load_stock_data(file_path)
            self.assertIsInstance(data, pd.DataFrame)
            self.assertFalse(data.empty)
            self.assertTrue(all(col in data.columns for col in ['Open', 'High', 'Low', 'Close', 'Volume']))
    
    def test_get_stock_info(self):
        """Test stock information retrieval"""
        for symbol in self.symbols:
            info = get_stock_info(symbol)
            self.assertIsNotNone(info)
            self.assertIsInstance(info, dict)
            self.assertTrue(all(key in info for key in ['symbol', 'name', 'sector', 'industry', 'market_cap', 'pe_ratio']))
    
    def test_load_stock_data(self):
        """Test stock data loading"""
        # Download data first
        download_stock_data(self.symbols, self.start_date, self.end_date, self.output_dir)
        
        # Test loading
        for symbol in self.symbols:
            file_path = f"{self.output_dir}/{symbol}.csv"
            data = load_stock_data(file_path)
            self.assertIsInstance(data, pd.DataFrame)
            self.assertFalse(data.empty)
            self.assertTrue(all(col in data.columns for col in ['Open', 'High', 'Low', 'Close', 'Volume']))

    def test_load_macro_data(self):
        """Test macroeconomic data loading"""
        # Load macroeconomic data
        macro_data = load_macro_data(start_date=self.start_date, end_date=self.end_date)
        
        # Check data format
        self.assertIsInstance(macro_data, pd.DataFrame)
        self.assertFalse(macro_data.empty)
        
        # Check if required macroeconomic indicators exist
        expected_indicators = ['GDP', 'CPI', 'UNRATE', 'FEDFUNDS']
        self.assertTrue(all(indicator in macro_data.columns for indicator in expected_indicators))
        
        # Check data time range (allowing for FRED data delay)
        min_date = pd.Timestamp(self.start_date) - pd.Timedelta(days=30)  # Allow 30 days margin
        max_date = pd.Timestamp(self.end_date) + pd.Timedelta(days=30)    # Allow 30 days margin
        self.assertTrue(macro_data.index.min() >= min_date, 
                       f"Earliest data date {macro_data.index.min()} is earlier than expected earliest date {min_date}")
        self.assertTrue(macro_data.index.max() <= max_date,
                       f"Latest data date {macro_data.index.max()} is later than expected latest date {max_date}")
        
        # Check data quality
        for indicator in expected_indicators:
            self.assertTrue(macro_data[indicator].notna().any(), 
                          f"Indicator {indicator} has no valid data")
            self.assertTrue(macro_data[indicator].notna().mean() > 0.5, 
                          f"Indicator {indicator} has too low valid data ratio")
        
        # Check if data is saved
        output_path = 'data/macro/macro_data.csv'
        self.assertTrue(os.path.exists(output_path))
        
if __name__ == '__main__':
    unittest.main() 