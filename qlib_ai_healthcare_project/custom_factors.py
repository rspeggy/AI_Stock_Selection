import numpy as np
import pandas as pd
import os
import pdb

class HealthcareFactors:
    """Healthcare industry specific factors"""

    @staticmethod
    def sector_volatility(high, low, close, window=20):
        tr = pd.DataFrame({
            'hl': high - low,
            'hc': abs(high - close.shift(1)),
            'lc': abs(low - close.shift(1))
        }).max(axis=1)
        volatility = tr.rolling(window=window).mean() / close
        return volatility

    @staticmethod
    def momentum_score(close, volume, window=20):
        price_momentum = close.pct_change(window)
        volume_momentum = volume.pct_change(window)
        momentum_score = (price_momentum + volume_momentum) / 2
        return momentum_score

    @staticmethod
    def healthcare_sentiment(close, volume, window=20):
        price_change = close.pct_change()
        volume_change = volume.pct_change()
        sentiment = (price_change * volume_change).rolling(window=window).mean()
        return sentiment

    @staticmethod
    def technical_indicators(high, low, close, volume, window=20):
        delta = close.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))

        exp1 = close.ewm(span=12, adjust=False).mean()
        exp2 = close.ewm(span=26, adjust=False).mean()
        macd = exp1 - exp2
        signal = macd.ewm(span=9, adjust=False).mean()

        ma = close.rolling(window=window).mean()
        std = close.rolling(window=window).std()
        upper_band = ma + (std * 2)
        lower_band = ma - (std * 2)

        return {
            'rsi': rsi,
            'macd': macd,
            'macd_signal': signal,
            'bollinger_upper': upper_band,
            'bollinger_lower': lower_band
        }

    @staticmethod
    def macro_impact_factor(stock_data, macro_data, window=20):
        """Calculate the impact of macroeconomic factors on stock returns"""
        # Calculate daily returns for stock
        stock_returns = stock_data['Close'].pct_change()
        
        # Calculate returns for each macro factor
        macro_returns = macro_data.pct_change()
        
        # Initialize correlation series
        correlations = pd.Series(0, index=stock_returns.index)
        
        # Calculate rolling correlation for each macro factor
        for col in macro_returns.columns:
            # Forward fill macro data to match stock data frequency
            macro_series = macro_returns[col].reindex(stock_returns.index, method='ffill')
            # Calculate rolling correlation
            corr = stock_returns.rolling(window=window).corr(macro_series)
            correlations += corr
        
        # Average the correlations across all macro factors
        average_corr = correlations / len(macro_returns.columns)
        
        return average_corr

    @staticmethod
    def interest_rate_sensitivity(stock_returns, fed_funds_rate, window=20):
        """
        Calculate stock's sensitivity to Federal Funds Rate changes
        """
        # Calculate rate changes using original data
        rate_changes = fed_funds_rate.pct_change()
        # Remove duplicate values (keep only when rate actually changes)
        rate_changes = rate_changes[rate_changes != 0].reindex(rate_changes.index)
        # Forward fill rate changes to match stock data frequency
        rate_changes = rate_changes.reindex(stock_returns.index, method='ffill')
        
        # Calculate rolling correlation
        sensitivity = stock_returns.rolling(window=window).corr(rate_changes)
        
        return sensitivity

    @staticmethod
    def inflation_impact(stock_returns, cpi_data, window=20):
        """
        Calculate stock's sensitivity to CPI changes
        """
        # Calculate CPI changes using original data
        cpi_changes = cpi_data.pct_change()
        # Remove duplicate values (keep only when CPI actually changes)
        cpi_changes = cpi_changes[cpi_changes != 0].reindex(cpi_changes.index)
        # Forward fill CPI changes to match stock data frequency
        cpi_changes = cpi_changes.reindex(stock_returns.index, method='ffill')
        
        # Calculate rolling correlation
        impact = stock_returns.rolling(window=window).corr(cpi_changes)
        
        return impact

    @staticmethod
    def economic_growth_correlation(stock_returns, gdp_data, window=20):
        """
        Calculate correlation between stock returns and GDP growth
        """
        # Calculate GDP growth using original data
        gdp_growth = gdp_data.pct_change()
        # Remove duplicate values (keep only when GDP actually changes)
        gdp_growth = gdp_growth[gdp_growth != 0].reindex(gdp_growth.index)
        # Forward fill GDP growth to match stock data frequency
        gdp_growth = gdp_growth.reindex(stock_returns.index, method='ffill')
        
        # Calculate rolling correlation
        correlation = stock_returns.rolling(window=window).corr(gdp_growth)
        
        return correlation

    @staticmethod
    def unemployment_impact(stock_returns, unemployment_rate, window=20):
        """
        Calculate stock's sensitivity to unemployment rate changes
        """
        # Calculate unemployment changes using original data
        unemployment_changes = unemployment_rate.diff()
        # Remove duplicate values (keep only when unemployment rate actually changes)
        unemployment_changes = unemployment_changes[unemployment_changes != 0].reindex(unemployment_changes.index)
        # Forward fill unemployment changes to match stock data frequency
        unemployment_changes = unemployment_changes.reindex(stock_returns.index, method='ffill')
        
        # Calculate rolling correlation
        impact = stock_returns.rolling(window=window).corr(unemployment_changes)
        
        return impact

    @staticmethod
    def sector_rotation_factor(stock_returns, sector_returns, window=20):
        """
        Calculate sector rotation factor based on relative performance
        Args:
            stock_returns: Daily stock returns
            sector_returns: Returns of other sectors
            window: Rolling window size
        Returns:
            Sector rotation score
        """
        # Calculate relative performance against other sectors
        relative_performance = stock_returns - sector_returns.mean(axis=1)
        rotation_score = relative_performance.rolling(window=window).mean()
        return rotation_score

def load_stock_data(stock_code):
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    file_path = os.path.join(BASE_DIR, 'data', 'raw', f'{stock_code}.csv')
    print(f"Attempting to load stock data: {file_path}")
    try:
        # Read first row as column names and replace the first column name with 'Date'
        columns = pd.read_csv(file_path, nrows=0).columns
        columns = ['Date'] + list(columns[1:])
        # Skip first two rows (Price and Ticker rows) and third row (empty row)
        data = pd.read_csv(file_path, skiprows=3, names=columns)
        # Set Date column as index
        data['Date'] = pd.to_datetime(data['Date'])
        data.set_index('Date', inplace=True)
        print(f"Successfully loaded stock data, shape: {data.shape}")
        print(f"Data columns: {data.columns.tolist()}")
        return data
    except Exception as e:
        print(f"Error loading stock data: {str(e)}")
        return None

def load_macro_data():
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    file_path = os.path.join(BASE_DIR, 'data', 'macro', 'macro_data_daily.csv')
    print(f"Attempting to load macro data: {file_path}")
    try:
        # Read the data with the first column as index
        data = pd.read_csv(file_path, index_col=0)
        # Convert index to datetime
        data.index = pd.to_datetime(data.index)
        
        # Ensure all columns are numeric
        for col in data.columns:
            data[col] = pd.to_numeric(data[col], errors='coerce')
        
        print(f"Successfully loaded macro data, shape: {data.shape}")
        print(f"Data columns: {data.columns.tolist()}")
        print(f"Date range: {data.index.min()} to {data.index.max()}")
        return data
    except Exception as e:
        print(f"Error loading macro data: {str(e)}")
        return None

def load_and_prepare_data(stock_code, start_date=None, end_date=None):
    print(f"\nPreparing data - Stock code: {stock_code}, Start date: {start_date}, End date: {end_date}")
    stock_data = load_stock_data(stock_code)
    macro_data = load_macro_data()

    if stock_data is None:
        print("Failed to load stock data")
        return None, None

    if not isinstance(stock_data.index, pd.DatetimeIndex):
        stock_data.index = pd.to_datetime(stock_data.index)

    if start_date:
        start_date = pd.to_datetime(start_date)
        stock_data = stock_data[stock_data.index >= start_date]
    if end_date:
        end_date = pd.to_datetime(end_date)
        stock_data = stock_data[stock_data.index <= end_date]

    if macro_data is not None:
        if not isinstance(macro_data.index, pd.DatetimeIndex):
            macro_data.index = pd.to_datetime(macro_data.index)
        
        # Print data info before alignment
        print("\nBefore alignment:")
        print(f"Stock data date range: {stock_data.index.min()} to {stock_data.index.max()}")
        print(f"Macro data date range: {macro_data.index.min()} to {macro_data.index.max()}")
        print(f"Stock data shape: {stock_data.shape}")
        print(f"Macro data shape: {macro_data.shape}")
        
        # Check data frequency
        print("\nChecking macro data frequency:")
        for col in macro_data.columns:
            # Count unique values
            unique_values = macro_data[col].nunique()
            # Count total rows
            total_rows = len(macro_data)
            print(f"{col}: {unique_values} unique values out of {total_rows} rows ({unique_values/total_rows*100:.2f}% unique)")
        
        # Filter macro data to match stock data date range
        macro_data = macro_data[
            (macro_data.index >= stock_data.index.min()) & 
            (macro_data.index <= stock_data.index.max())
        ]
        
        # Print data info after filtering
        print("\nAfter filtering:")
        print(f"Macro data shape: {macro_data.shape}")
        print(f"Macro data columns: {macro_data.columns.tolist()}")
        print(f"Macro data sample (first 5 rows):")
        print(macro_data.head())
        print("\nMacro data sample (last 5 rows):")
        print(macro_data.tail())
        
        # Check for missing values
        missing_values = macro_data.isnull().sum()
        print("\nMissing values in macro data:")
        print(missing_values)

    print(f"\nData preparation completed - Stock data shape: {stock_data.shape}, Macro data shape: {macro_data.shape if macro_data is not None else 'None'}")
    return stock_data, macro_data

def create_custom_factors(data, macro_data=None):
    factors = pd.DataFrame(index=data.index)

    # Original factors
    factors['sector_volatility'] = HealthcareFactors.sector_volatility(data['High'], data['Low'], data['Close'])
    factors['momentum_score'] = HealthcareFactors.momentum_score(data['Close'], data['Volume'])
    factors['healthcare_sentiment'] = HealthcareFactors.healthcare_sentiment(data['Close'], data['Volume'])

    tech_indicators = HealthcareFactors.technical_indicators(data['High'], data['Low'], data['Close'], data['Volume'])
    for name, indicator in tech_indicators.items():
        factors[name] = indicator

    # Macro impact factor
    if macro_data is not None:
        print("\nProcessing macro impact factor...")
        macro_impact_series = HealthcareFactors.macro_impact_factor(data, macro_data)
        factors['macro_impact'] = macro_impact_series

        # Print factor statistics
        print("\nFactor statistics:")
        for col in factors.columns:
            non_zero = factors[col].count()
            total = len(factors[col])
            print(f"{col}: {non_zero}/{total} non-zero values ({non_zero/total*100:.2f}%)")

    return factors.reindex(data.index).fillna(0)

def process_all_stocks(start_date=None, end_date=None):
    """Process all stock data and generate factors"""
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    raw_dir = os.path.join(BASE_DIR, 'data', 'raw')
    factors_dir = os.path.join(BASE_DIR, 'data', 'factors')
    os.makedirs(factors_dir, exist_ok=True)

    # Get all CSV files
    stock_files = [f for f in os.listdir(raw_dir) if f.endswith('.csv')]
    total_stocks = len(stock_files)
    
    print(f"\nStarting to process factors for {total_stocks} stocks...")
    
    for idx, stock_file in enumerate(stock_files, 1):
        stock_code = stock_file.replace('.csv', '')
        print(f"\nProgress: [{idx}/{total_stocks}] Processing stock: {stock_code}")
        
        try:
            stock_data, macro_data = load_and_prepare_data(stock_code, start_date, end_date)
            
            if stock_data is not None:
                factors = create_custom_factors(stock_data, macro_data)
                output_file = os.path.join(factors_dir, f'{stock_code}_factors.csv')
                factors.to_csv(output_file)
                print(f"✓ Successfully generated factors and saved to: {output_file}")
            else:
                print(f"✗ Unable to process data for stock {stock_code}")
                
        except Exception as e:
            print(f"✗ Error processing stock {stock_code}: {str(e)}")
            continue
    
    print("\nAll stock factor processing completed!")

def load_factors(stock_code: str, start_date=None, end_date=None) -> pd.DataFrame:
    """
    直接读取已生成的因子数据
    
    Args:
        stock_code (str): 股票代码
        start_date (str, optional): 开始日期
        end_date (str, optional): 结束日期
        
    Returns:
        pd.DataFrame: 因子数据
    """
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    factors_path = os.path.join(BASE_DIR, 'data', 'factors', f'{stock_code}_factors.csv')
    
    try:
        # 读取因子数据
        factors = pd.read_csv(factors_path, index_col=0)
        factors.index = pd.to_datetime(factors.index)
        
        # 如果指定了日期范围，进行过滤
        if start_date:
            start_date = pd.to_datetime(start_date)
            factors = factors[factors.index >= start_date]
        if end_date:
            end_date = pd.to_datetime(end_date)
            factors = factors[factors.index <= end_date]
            
        print(f"成功加载因子数据: {factors_path}")
        print(f"因子数据形状: {factors.shape}")
        print(f"因子列表: {factors.columns.tolist()}")
        
        return factors
        
    except Exception as e:
        print(f"加载因子数据失败: {str(e)}")
        return None

if __name__ == "__main__":
    start_date = "2019-01-01"
    end_date = "2024-01-01"
    
    # 处理所有股票
    process_all_stocks(start_date, end_date)
