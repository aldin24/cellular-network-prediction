import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import kagglehub
import os

plt.style.use('ggplot')
sns.set_palette("husl")

# Download and load the data
path = kagglehub.dataset_download("suraj520/cellular-network-analysis-dataset")
files = os.listdir(path)
csv_files = [f for f in files if f.endswith(".csv")]
file_path = os.path.join(path, csv_files[0])
df = pd.read_csv(file_path)


# Correlation heatmap visualization
# Select only numeric columns
numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()

# Calculate correlation matrix
corr_matrix = df[numeric_cols].corr()

# Create full correlation heatmap
plt.figure(figsize=(12, 10))
sns.heatmap(corr_matrix, annot=True, fmt='.2f', cmap='coolwarm', center=0,
            square=True, linewidths=1, cbar_kws={"shrink": 0.8})
plt.title('Correlation Matrix - All Numeric Features', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.show()

# Focused view: Correlations with Data Throughput
target = 'Data Throughput (Mbps)'
if target in numeric_cols:
    # Get correlations with target and sort
    correlations_with_target = df[numeric_cols].corr()[target].sort_values(ascending=False)
    # Remove the target itself (it will be 1.0)
    correlations_with_target = correlations_with_target[correlations_with_target.index != target]
    
    # Create bar plot
    plt.figure(figsize=(10, 6))
    correlations_with_target.plot(kind='barh', color=['green' if x > 0 else 'red' for x in correlations_with_target.values])
    plt.xlabel('Correlation with Data Throughput', fontsize=12)
    plt.title('Feature Correlations with Data Throughput (Mbps)', fontsize=14, fontweight='bold')
    plt.axvline(x=0, color='black', linestyle='--', linewidth=0.8)
    plt.tight_layout()
    plt.show()
    
    # Print the correlations
    print("\nCorrelations with Data Throughput:")
    print(correlations_with_target)

