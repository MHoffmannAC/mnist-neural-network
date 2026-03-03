
from pytrends import dailydata


def fetch_10_year_daily_data(keyword, start_year, end_year, geo="US"):
    """Fetches daily Google Trends data by stitching together smaller windows
    to maintain daily resolution over a long period.
    """
    print(f"Starting data extraction for: {keyword}")
    print(f"Range: {start_year} to {end_year} ({geo})")

    try:
        # get_daily_data handles the heavy lifting:
        # 1. Fetches monthly data for the entire period as a baseline.
        # 2. Fetches daily data in small overlapping chunks.
        # 3. Scales the daily data to match the monthly baseline.
        df = dailydata.get_daily_data(
            word=keyword,
            start_year=start_year,
            start_mon=1,
            stop_year=end_year,
            stop_mon=12,
            geo=geo,
            verbose=True,
            wait_time=60.0, # Increase this if you get 429 (Too Many Requests) errors
        )

        # The resulting dataframe contains:
        # - '{keyword}_unscaled': The raw daily values from the chunks
        # - '{keyword}_monthly': The monthly baseline values
        # - '{keyword}': The final, stitched, and rescaled daily data (0-100)

        # Clean up the dataframe
        final_df = df[[keyword]].copy()
        final_df.index.name = "Date"
        final_df.columns = ["Trend_Score"]

        return final_df

    except Exception as e:
        print(f"An error occurred: {e}")
        return None

if __name__ == "__main__":
    # --- CONFIGURATION ---
    SEARCH_TERM = "NFL"
    START = 2004
    END = 2026
    LOCATION = ""  # Empty string for Worldwide, or 'US', 'GB', etc.
    # ---------------------

    results = fetch_10_year_daily_data(SEARCH_TERM, START, END, LOCATION)

    if results is not None:
        filename = f"{SEARCH_TERM.replace(' ', '_')}_daily_trends.csv"
        results.to_csv(filename)
        print(f"\nSuccess! Data saved to {filename}")
        print(results.head())
        print("...")
        print(results.tail())
