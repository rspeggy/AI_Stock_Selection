import os
import pandas as pd
from dotenv import load_dotenv
from fredapi import Fred

def convert_quarterly_to_daily(df):
    """
    Convert quarterly data to daily data by forward filling values
    
    Args:
        df (pd.DataFrame): DataFrame with quarterly data
        
    Returns:
        pd.DataFrame: DataFrame with daily data
    """
    # Create a daily date range
    daily_dates = pd.date_range(start=df.index.min(), end=df.index.max(), freq='D')
    
    # Reindex the dataframe to daily frequency and forward fill values
    daily_df = df.reindex(daily_dates, method='ffill')
    
    return daily_df

def load_macro_data(start_date='2015-01-01', end_date=None):
    """
    Load macroeconomic data from FRED
    
    Args:
        start_date (str): Start date in 'YYYY-MM-DD' format
        end_date (str): End date in 'YYYY-MM-DD' format, defaults to today
        
    Returns:
        pd.DataFrame: Macroeconomic data
    """
    # Load environment variables
    load_dotenv()
    
    # Get API key
    FRED_API_KEY = os.getenv('FRED_API_KEY')
    if not FRED_API_KEY:
        raise ValueError("Please set FRED_API_KEY in .env file")
    
    # Initialize FRED client
    fred = Fred(api_key=FRED_API_KEY)
    
    # Set end date
    if end_date is None:
        end_date = pd.Timestamp.today().strftime('%Y-%m-%d')
    
    # Convert dates to datetime for comparison
    start_date = pd.Timestamp(start_date)
    end_date = pd.Timestamp(end_date)
    
    # Macroeconomic series to fetch
    macro_series = {
        'GDP': 'GDP',                   # US Gross Domestic Product
        'CPI': 'CPIAUCSL',             # Consumer Price Index
        'UNRATE': 'UNRATE',            # Unemployment Rate
        'FEDFUNDS': 'FEDFUNDS',        # Federal Funds Rate
    }
    
    # Download data
    macro_data = pd.DataFrame()
    for name, series_id in macro_series.items():
        try:
            data = fred.get_series(series_id, start_date=start_date, end_date=end_date)
            # 确保数据在指定时间范围内
            data = data.loc[start_date:end_date]
            macro_data[name] = data
            print(f"Successfully retrieved {name} data")
        except Exception as e:
            print(f"Error retrieving {name} data: {str(e)}")
    
    # Process data
    macro_data.index.name = 'Date'
    macro_data = macro_data.sort_index()
    
    # 确保数据在指定时间范围内
    macro_data = macro_data.loc[start_date:end_date]
    
    # Convert quarterly data to daily data
    daily_macro_data = convert_quarterly_to_daily(macro_data)
    
    # Save data
    output_path = 'data/macro/macro_data_daily.csv'
    os.makedirs(os.path.dirname(output_path), exist_ok=True)  # 确保目录存在
    daily_macro_data.to_csv(output_path)
    print(f"\nDaily macroeconomic data saved to: {output_path}")
    
    return daily_macro_data

if __name__ == "__main__":
    # Example usage
    try:
        data = load_macro_data()
        print("\nData preview:")
        print(data.head())
    except Exception as e:
        print(f"Error: {str(e)}") 