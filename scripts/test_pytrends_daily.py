
import pandas as pd
from pytrends.request import TrendReq
from datetime import date, timedelta

def test_daily_granularity():
    pytrends = TrendReq(hl='en-US', tz=360)
    keyword = "NFL"
    
    # Test for a short period (e.g., 90 days) which should return daily data
    end_date = date.today()
    start_date = end_date - timedelta(days=89) # 90 days inclusive
    
    timeframe = f"{start_date.isoformat()} {end_date.isoformat()}"
    print(f"Fetching data for '{keyword}' for timeframe: {timeframe}")

    try:
        pytrends.build_payload([keyword], cat=0, timeframe=timeframe, geo='', gprop='')
        data = pytrends.interest_over_time()

        if data.empty:
            print("No data fetched.")
            return

        if 'isPartial' in data.columns:
            data = data.drop(columns=['isPartial'])

        print("\nSample of fetched data:")
        print(data.head())
        print(data.tail())
        print(f"\nNumber of data points: {len(data)}")
        print(f"Expected number of data points for {timeframe}: {(end_date - start_date).days + 1}")

    except Exception as e:
        print(f"Error fetching data: {e}")

if __name__ == "__main__":
    test_daily_granularity()
