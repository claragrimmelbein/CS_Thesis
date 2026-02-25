import pandas as pd
import matplotlib.pyplot as plt

# load results from evaluate.py
df = pd.read_csv('model_results_comparison.csv')

# calculate the residuals
# residual = actual value - predicted value
df['Residual'] = df['Actual_Watch_Time'] - df['Predicted_Watch_Time']

# create the plot
plt.figure(figsize=(10, 6))
plt.scatter(df['Predicted_Watch_Time'], df['Residual'], alpha=0.5, color='royalblue', edgecolors='k')

# cdd a horizontal line at 0 
plt.axhline(y=0, color='red', linestyle='--', linewidth=2)

# labels for thesis
plt.title('Residual Plot: Transformer Model Performance', fontsize=14)
plt.xlabel('Predicted Watch Time (Seconds)', fontsize=12)
plt.ylabel('Residual (Error Size)', fontsize=12)
plt.grid(True, linestyle=':', alpha=0.7)

# cave the figure
plt.savefig('transformer_residual_plot.png', dpi=300)
print("Success!'")
plt.show()