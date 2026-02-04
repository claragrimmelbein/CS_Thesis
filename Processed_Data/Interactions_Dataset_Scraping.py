import json
import pandas as pd
import os

# load data
file_path = 'user1_data_tiktok.json'
with open(file_path, 'r', encoding='utf-8') as f:
    data = json.load(f)

# extract watch history 
watch_data = data.get('Your Activity', {}).get('Watch History', {}).get('VideoList', [])
df_watch = pd.DataFrame(watch_data)
df_watch.to_csv('tiktok_watch_history.csv', index=False)

# extract social network (Followers and Following)
followers = data.get('Profile And Settings', {}).get('Follower', {}).get('FansList', [])
following = data.get('Profile And Settings', {}).get('Following', {}).get('FollowingList', [])
pd.DataFrame(followers).to_csv('tiktok_followers.csv', index=False)
pd.DataFrame(following).to_csv('tiktok_following.csv', index=False)

# extract comments
comments = data.get('Comment', {}).get('Comments', {}).get('CommentsList', [])
pd.DataFrame(comments).to_csv('tiktok_comments.csv', index=False)

# extract likes & favorites
likes = data.get('Likes and Favorites', {}).get('Like List', {}).get('ItemFavoriteList', [])
favs = data.get('Likes and Favorites', {}).get('Favorite Videos', {}).get('FavoriteVideoList', [])
pd.DataFrame(likes).to_csv('tiktok_likes.csv', index=False)
pd.DataFrame(favs).to_csv('tiktok_favorites.csv', index=False)

print("csv finished)
