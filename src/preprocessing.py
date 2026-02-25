import pandas as pd
import numpy as np
import torch # std lib for transformers

def preprocess_data():
    # load in 
    interactions = pd.read_csv('Interactions_Dataset.csv')
    watch_time = pd.read_csv('tiktok_watch_time_estimated.csv')
    
    # standardize time
    interactions['Timestamp'] = pd.to_datetime(interactions['Timestamp'])
    watch_time['Date'] = pd.to_datetime(watch_time['Date'])
    
    # cyclical time encoding
    # use sin/cos so the model knows 11:59 PM is close to 12:01 AM
    def encode_cyclical(df, col, max_val):
        df[col + '_sin'] = np.sin(2 * np.pi * df[col]/max_val)
        df[col + '_cos'] = np.cos(2 * np.pi * df[col]/max_val)
        return df

    watch_time['hour'] = watch_time['Date'].dt.hour
    watch_time = encode_cyclical(watch_time, 'hour', 24)
    
    # categorical Encoding
    # map 'Like', 'Favorite', 'Comment' to numbers
    type_map = {'Like': 0, 'Favorite': 1, 'Comment': 2, 'Watch': 3}
    # (For this demo, we assume general watch history is type 'Watch')
    
    # create sequence windows
    # transformer needs a "context window" (ie the last 10 videos)
    def create_sequences(data, window_size=10):
        sequences = []
        targets = []
        # use watch time as the target to predict
        values = data['Estimated_Watch_Time_Sec'].fillna(0).values
        
        for i in range(len(values) - window_size):
            window = values[i : i + window_size]
            target = values[i + window_size]
            sequences.append(window)
            targets.append(target)
            
        return torch.tensor(sequences, dtype=torch.float32), torch.tensor(targets, dtype=torch.float32)

    # sort watch time by date
    watch_time = watch_time.sort_values('Date')
    X, y = create_sequences(watch_time)
    
    print(f"Prepared {X.shape[0]} sequences for the Transformer.")
    return X, y

if __name__ == "__main__":
    X, y = preprocess_data()
    # save prepared tensors for the model script
    torch.save(X, 'X_train.pt')
    torch.save(y, 'y_train.pt')