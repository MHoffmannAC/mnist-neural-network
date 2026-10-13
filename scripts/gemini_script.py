import time
import datetime
import random
import pandas as pd
from pytrends.request import TrendReq

def get_daily_trends(keyword: str, start_year: int = 2006, geo: str = "") -> pd.DataFrame:
    """
    Fetches daily Google Trends data across long timeframes by requesting 
    overlapping 90-day blocks and rescaling them, with a 5-minute pause 
    between requests to respect rate limits.
    """
    pytrends = TrendReq(hl='en-US', tz=360)
    
    start_date = datetime.date(start_year, 1, 1)
    end_date = datetime.date.today()
    
    # 1. Generate 90-day overlapping intervals
    delta_days = 90
    current_start = start_date
    date_ranges = []
    
    while current_start < end_date:
        current_end = min(current_start + datetime.timedelta(days=delta_days), end_date)
        timeframe = f"{current_start.strftime('%Y-%m-%d')} {current_end.strftime('%Y-%m-%d')}"
        date_ranges.append((timeframe, current_start, current_end))
        # Move forward with 10 days overlap
        current_start = current_end - datetime.timedelta(days=10)
        
        if current_end >= end_date:
            break

    full_df = pd.DataFrame()
    last_overlap_val = None
    total_batches = len(date_ranges)

    print(f"Fetching {total_batches} daily batches for '{keyword}'.")
    print(f"Note: Pausing ~5 minutes between requests to prevent HTTP 429 blockages.\n")

    for idx, (tf, s_date, e_date) in enumerate(date_ranges):
        print(f"[{idx+1}/{total_batches}] Fetching period: {tf}...")
        
        try:
            pytrends.build_payload([keyword], cat=0, timeframe=tf, geo=geo)
            batch_df = pytrends.interest_over_time()
        except Exception as e:
            print(f"   Warning: Request failed ({e}). Retrying after pause...")
            time.sleep(300)
            continue
        
        if batch_df.empty or keyword not in batch_df.columns:
            print("   No data returned for this chunk.")
            continue
            
        batch_df = batch_df[[keyword]]
        
        # Rescale current chunk relative to previous chunk's overlapping score
        if last_overlap_val is not None and not full_df.empty:
            overlap_date = batch_df.index[0]
            new_overlap_val = batch_df.loc[overlap_date, keyword]
            
            if new_overlap_val > 0:
                scale_factor = last_overlap_val / new_overlap_val
                batch_df[keyword] = batch_df[keyword] * scale_factor

        # Append non-overlapping data
        if full_df.empty:
            full_df = batch_df
        else:
            new_rows = batch_df[batch_df.index > full_df.index[-1]]
            full_df = pd.concat([full_df, new_rows])
            
        last_overlap_val = batch_df.iloc[-1][keyword]
        
        # 5-minute pause (300 seconds) + slight random variance
        if idx < total_batches - 1:
            pause_time = 300 + random.uniform(1, 10)
            print(f"   Done. Waiting {pause_time / 60:.1f} minutes before next request...\n")
            time.sleep(pause_time)

    # Re-normalize entire series between 0 and 100
    if not full_df.empty:
        max_val = full_df[keyword].max()
        if max_val > 0:
            full_df[keyword] = (full_df[keyword] / max_val) * 100

    filename = f"{keyword.lower().replace(' ', '_')}_daily_trends.csv"
    full_df.to_csv(filename)
    print(f"\nCompleted! Saved daily search scores to {filename}")
    
    return full_df

if __name__ == "__main__":
    df_daily = get_daily_trends("NFL", start_year=2000)