import numpy as np
import pandas as pd

# Set random seed for reproducibility
np.random.seed(42)

# Define the number of samples
num_samples = 1000 #500 false; #500 positive 

# Generate synthetic features for non-fentanyl samples
non_fentanyl_features = np.random.normal(loc=5, scale=2, size=(num_samples // 2, 5))

# Generate synthetic features for fentanyl samples
fentanyl_features = np.random.normal(loc=8, scale=2, size=(num_samples // 2, 5))

# Create labels (0 for non-fentanyl, 1 for fentanyl)
labels = np.concatenate([np.zeros(num_samples // 2), np.ones(num_samples // 2)])

# Combine features and labels into a DataFrame
columns = ['Feature1', 'Feature2', 'Feature3', 'Feature4', 'Feature5']
data = np.vstack([non_fentanyl_features, fentanyl_features])
df = pd.DataFrame(data, columns=columns)
df['Label'] = labels

# Shuffle the dataset
df = df.sample(frac=1, random_state=42).reset_index(drop=True)

# Save the synthetic dataset to a CSV file
df.to_csv('synthetic_fentanyl_dataset.csv', index=False)
