
import pandas as pd
from pytrends.request import TrendReq
from datetime import date, timedelta
import time
import sys

def get_google_trends_data(keyword, start_date_str, end_date_str, output_file='nfl_google_trends.csv'):
    """
    Fetches daily Google Trends data for a given keyword and time range,
    handling data normalization across multiple requests.

    Args:
        keyword (str): The search term (e.g., "NFL").
        start_date_str (str): Start date in 'YYYY-MM-DD' format.
        end_date_str (str): End date in 'YYYY-MM-DD' format.
        output_file (str): Name of the CSV file to save the data.
    """
    pytrends = TrendReq(hl='en-US', tz=360)
    
    start_date = date.fromisoformat(start_date_str)
    end_date = date.fromisoformat(end_date_str)

    all_data = pd.DataFrame()
    current_start = start_date

    # Define chunk size and overlap
    # pytrends typically allows a max of 270 days for daily data.
    # A chunk size of 269 days with a 30-day overlap is a good balance for daily granularity.
    timeframe_chunk_days = 269
    overlap_days = 30 # Overlap to ensure consistent scaling

    print(f"Fetching Google Trends data for '{keyword}' from {start_date_str} to {end_date_str}")

    while current_start <= end_date:
        chunk_end = min(current_start + timedelta(days=timeframe_chunk_days) - timedelta(days=1), end_date)
        
        # Adjust chunk_start to include overlap if it's not the very first chunk
        if not all_data.empty and current_start > start_date:
            # For subsequent chunks, start earlier to create an overlap with the previous chunk
            # Ensure we don't go before the actual start_date
            adjusted_current_start = max(current_start - timedelta(days=overlap_days), start_date)
            timeframe = f"{adjusted_current_start.isoformat()} {chunk_end.isoformat()}"
        else:
            timeframe = f"{current_start.isoformat()} {chunk_end.isoformat()}"
        
        print(f"  Fetching chunk: {timeframe}")

        try:
            pytrends.build_payload([keyword], cat=0, timeframe=timeframe, geo='', gprop='')
            data = pytrends.interest_over_time()
            
            if data.empty or keyword not in data.columns:
                print(f"    No data or keyword '{keyword}' found for this chunk. Skipping.")
                current_start = chunk_end + timedelta(days=1)
                continue

            # Drop the 'isPartial' column if it exists
            if 'isPartial' in data.columns:
                data = data.drop(columns=['isPartial'])
            
            # If this is the first chunk, just add it
            if all_data.empty:
                all_data = data
            else:
                # Re-normalize data based on overlap
                # Find the overlapping period
                overlap_period = all_data.index.intersection(data.index)

                if not overlap_period.empty:
                    # Calculate scaling factor from the overlapping period
                    # Avoid division by zero for cases where overlap trend is zero
                    all_data_overlap_mean = all_data.loc[overlap_period, keyword].replace(0, 0.001).mean()
                    data_overlap_mean = data.loc[overlap_period, keyword].replace(0, 0.001).mean()
                    
                    if data_overlap_mean > 0:
                        scaling_factor = all_data_overlap_mean / data_overlap_mean
                        data[keyword] = data[keyword] * scaling_factor
                        print(f"    Scaled chunk by factor: {scaling_factor:.4f}")
                    else:
                        print("    Overlap data in current chunk is zero, unable to scale. Appending directly.")
                else:
                    print("    No overlap found, appending chunk directly.")
                
                # Append the new data, avoiding duplicates in the overlap period
                all_data = pd.concat([all_data, data[~data.index.isin(all_data.index)]])

            print(f"  Chunk fetched successfully. Total data points: {len(all_data)}")

        except Exception as e:
            print(f"    Error fetching data for chunk {timeframe}: {e}")
            print("    Waiting for 60 seconds before retrying...")
            time.sleep(60) # Wait before retrying on error
            continue # Retry the same chunk

        current_start = chunk_end + timedelta(days=1) # Move to the next day after the chunk_end
        time.sleep(1) # Be polite to the API

    if all_data.empty:
        print("No data was fetched. Exiting.")
        return

    # Sort by date and clean up
    all_data = all_data.sort_index()
    all_data.index.name = 'ds'
    all_data = all_data.reset_index()
    all_data = all_data[['ds', keyword]] # Ensure order and only keyword column
    all_data = all_data.rename(columns={keyword: 'y'})

    print(f"\nData fetching complete. Total entries: {len(all_data)}")
    print("Sample data:")
    print(all_data.head())
    print(all_data.tail())

    all_data.to_csv(output_file, index=False)
    print(f"Data saved to {output_file}")

if __name__ == "__main__":
    KEYWORD = "NFL"
    START_DATE = "2004-01-01"
    # Use today's date for the end date
    END_DATE = date.today().isoformat()
    
    # Check if the script is being run as part of a scheduled job or continuously
    # If so, append to the existing file rather than overwriting
    # This example simply overwrites, but for continuous updates, you'd load the existing
    # data and fetch only new data
    
    # Call the function to get and save the data
    get_google_trends_data(KEYWORD, START_DATE, END_DATE)
