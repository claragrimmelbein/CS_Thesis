import torch
import matplotlib.pyplot as plt
import numpy as np

# load test tensors 
y_test = torch.load('y_test.pt').numpy()

# define binging vs long-form
# TikToks < 15s are binge bursts and > 60s are long-form
binge_threshold = 15
long_form_threshold = 60

binges = y_test[y_test < binge_threshold]
long_form = y_test[y_test > long_form_threshold]

# print the stats
print(f"Total Test Samples: {len(y_test)}")
print(f"Binge Sequences (<15s): {len(binges)} ({len(binges)/len(y_test)*100:.1f}%)")
print(f"Long-form Sequences (>60s): {len(long_form)} ({len(long_form)/len(y_test)*100:.1f}%)")

# visualize the distribution
plt.figure(figsize=(10, 6))
plt.hist(y_test, bins=50, color='skyblue', edgecolor='black', range=(0, 150))
plt.axvline(binge_threshold, color='red', linestyle='dashed', label='Binge Threshold')
plt.title('Distribution of Watch Times in Test Set')
plt.xlabel('Seconds Watched')
plt.ylabel('Frequency')
plt.legend()
plt.savefig('watch_time_distribution.png')
print("Graph saved as watch_time_distribution.png")