import pandas as pd
import numpy as np
import torch
from sklearn.model_selection import train_test_split

def preprocess_data(split_mode='chronological', train_size=0.2):
    # load data
    watch_time = pd.read_csv('Processed_Data/tiktok_watch_time_estimated.csv')
    watch_time['Date'] = pd.to_datetime(watch_time['Date'])
    watch_time = watch_time.sort_values('Date') 

    # calculation of latency ie pausing between videos
    watch_time['latency'] = watch_time['Date'].diff().dt.total_seconds().fillna(0)


    # cyclical encoding based on hour
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
    def create_sequences(df, max_window=10):
        # pylling watch time, latencies and hour features into 1D array
        watch_values = df['Estimated_Watch_Time_Sec'].fillna(0).values
        latencies = df['latency'].values
        hours_sin = df['hour_sin'].values
            
        sequences = []
        targets = []

        # starting at index 1, second video prediction based on first
        for i in range(1,len(watch_values)):
            # determines window size 
            start_idx = max(0, i - max_window)

            # combining featuresa into multi-feature vector for seach video
            # each step in sequence is watch time , latency, hoursin
            seq_slice = np.column_stack((
                watch_values[start_idx:i],
                latencies[start_idx:i],
                hours_sin[start_idx:i]
            ))

            # if sequence shorter than max_window pad w 0's
            # I think this is supposed to help w keeping consistent tensor shape for the transformer?
            if len(seq_slice) < max_window:
                padding = np.zeros((max_window - len(seq_slice), 3))
                seq_slice = np.vstack((padding, seq_slice))

            sequences.append(seq_slice)
            targets.append(watch_values[i])

  
        return torch.tensor(np.array(sequences), dtype=torch.float32), \
               torch.tensor(np.array(targets), dtype=torch.float32)

    # generate tensors
    X_train, y_train = create_sequences(train_df)
    X_test, y_test = create_sequences(test_df)

    print(f"Split Mode: {split_mode}")
    print(f"Features encoded: [WatchTime, Latency, Hour_Sin]")
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