import json
import pandas as pd

file_name = 'user1_data_tiktok.json'
with open(file_name, 'r', encoding='utf-8') as f:
    data = json.load(f)

watch_history = data.get('Your Activity', {}).get('Watch History', {}).get('VideoList', [])

if watch_history:
    df = pd.DataFrame(watch_history)
    df['Date'] = pd.to_datetime(df['Date'])
    
    # 2. Sort by date to find the time between videos
    df = df.sort_values('Date')
    
    # 3. Calculate seconds until the next video
    df['Time_Until_Next_Sec'] = (df['Date'].shift(-1) - df['Date']).dt.total_seconds()
    
    # 4. Filter out long gaps (sessions ending)
    # If gap is > 600 seconds (10 mins), it's likely a new session, not a long video
    df['Estimated_Watch_Time_Sec'] = df['Time_Until_Next_Sec'].apply(lambda x: x if x < 600 else None)
    
    # 5. Save the result (most recent first)
    df = df.sort_values('Date', ascending=False)
    df.to_csv('tiktok_watch_time_estimated.csv', index=False)
    
    print(f"Done! Processed {len(df)} videos.")
    print(f"Average watch time: {df['Estimated_Watch_Time_Sec'].mean():.2f} seconds.")
else:
    print("No watch history found.")