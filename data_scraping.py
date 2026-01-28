import pandas as pd
import json

# load the JSON file from TikTok
# waiting on data to be sent from tiktok
with open('user_data.json', 'r', encoding='utf-8') as f:
    data = json.load(f)

# extract browsing history
history_list = data['Activity']['Video Browsing History']['VideoList']
df_history = pd.DataFrame(history_list)

# extract likes
like_list = data['Activity']['Like List']['ItemFavoriteList']
df_likes = pd.DataFrame(like_list)

# create the "attention label"
# set of URLs that were liked for fast lookup
liked_urls = set(df_likes['Link'].tolist())

def calculate_attention_proxy(video_url):
    if video_url in liked_urls:
        # high Attention (engaged)
        return 1 
    else:
        # low attention (passive/skipped)
        return 0 

# apply label
df_history['attention_label'] = df_history['Link'].apply(calculate_attention_proxy)

print(f"Total Videos Watched: {len(df_history)}")
print(f"High Attention Interactions: {df_history['attention_label'].sum()}")
print(df_history.head())
