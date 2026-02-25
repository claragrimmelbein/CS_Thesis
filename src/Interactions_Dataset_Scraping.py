import json
import pandas as pd
import os

file_name = 'user1_data_tiktok.json'

if not os.path.exists(file_name):
    print(f"ERROR: The file '{file_name}' was not found in this folder.")
    print("Files currently in this folder:", os.listdir('.'))
else:
    try:
        with open(file_name, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        # extract data
        comments = data.get('Comment', {}).get('Comments', {}).get('CommentsList', [])
        likes = data.get('Likes and Favorites', {}).get('Like List', {}).get('ItemFavoriteList', [])
        favorites = data.get('Likes and Favorites', {}).get('Favorite Videos', {}).get('FavoriteVideoList', [])

        # create dataFrames
        df_c = pd.DataFrame(comments).rename(columns={'date': 'Timestamp', 'comment': 'Content', 'url': 'Link'})
        df_c['Type'] = 'Comment'
        
        df_l = pd.DataFrame(likes).rename(columns={'date': 'Timestamp', 'link': 'Link'})
        df_l['Type'] = 'Like'
        
        df_f = pd.DataFrame(favorites).rename(columns={'Date': 'Timestamp', 'Link': 'Link'})
        df_f['Type'] = 'Favorite'

        # merge and save
        df_master = pd.concat([df_c, df_l, df_f], ignore_index=True)
        df_master.to_csv('Interactions_Dataset.csv', index=False)
        
        print(f"Success! Created 'Interactions_Dataset.csv' with {len(df_master)} rows.")

    except json.JSONDecodeError:
        print(f"ERROR: '{file_name}' is not a valid JSON file. It might be empty or corrupted.")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")