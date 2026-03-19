import pandas as pd
import numpy as np
import torch
from sklearn.model_selection import train_test_split

def preprocess_data(split_mode='chronological', train_size=0.2):
    # Load Data
    watch_time = pd.read_csv('Processed_Data/tiktok_watch_time_estimated.csv')
    watch_time['Date'] = pd.to_datetime(watch_time['Date'])
    watch_time = watch_time.sort_values('Date') 

    # cyclical encoding 
    watch_time['hour'] = watch_time['Date'].dt.hour
    def encode_cyclical(df, col, max_val):
        df[col + '_sin'] = np.sin(2 * np.pi * df[col]/max_val)
        df[col + '_cos'] = np.cos(2 * np.pi * df[col]/max_val)
        return df
    watch_time = encode_cyclical(watch_time, 'hour', 24)

    # SPLIT DATA FIRST
    if split_mode == 'chronological':
        split_idx = int(len(watch_time) * train_size)
        train_df = watch_time.iloc[:split_idx]
        test_df = watch_time.iloc[split_idx:]
    else:
        # standard random split 
        train_df, test_df = train_test_split(watch_time, train_size=train_size, shuffle=True)

    # sequence windows helper
    def create_sequences(df, window_size=10):
        values = df['Estimated_Watch_Time_Sec'].fillna(0).values
        if len(values) <= window_size:
            return torch.tensor([]), torch.tensor([])
            
        sequences = []
        targets = []
        for i in range(len(values) - window_size):
            sequences.append(values[i : i + window_size])
            targets.append(values[i + window_size])
            
        return torch.tensor(np.array(sequences), dtype=torch.float32), \
               torch.tensor(np.array(targets), dtype=torch.float32)

    # generate tensors
    X_train, y_train = create_sequences(train_df)
    X_test, y_test = create_sequences(test_df)

    print(f"Split Mode: {split_mode}")
    print(f"Train sequences: {X_train.shape[0]} | Test sequences: {X_test.shape[0]}")
    
    return X_train, y_train, X_test, y_test

if __name__ == "__main__":
    # testing out 20% training
    X_train, y_train, X_test, y_test = preprocess_data(split_mode='chronological', train_size=0.2)
    
    # save all four evaluate.py --> to use test 
    torch.save(X_train, 'X_train.pt')
    torch.save(y_train, 'y_train.pt')
    torch.save(X_test, 'X_test.pt')
    torch.save(y_test, 'y_test.pt')