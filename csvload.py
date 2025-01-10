import pandas as pd
import numpy as np

# Configuration
NUM_FEATURES = 128
NUM_TARGETS = 64
NUM_SAMPLES = 1000  # Number of samples in the dataset

# Generate random feature data (128 features per sample)
features = np.random.rand(NUM_SAMPLES, NUM_FEATURES)

# Generate random target data (64 targets per sample) with values between -1 and 1
targets = np.random.uniform(low=-1.0, high=1.0, size=(NUM_SAMPLES, NUM_TARGETS))

# Combine features and targets into one dataset
data = np.hstack((features, targets))

# Create column names
feature_columns = [f'feature{i+1}' for i in range(NUM_FEATURES)]
target_columns = [f'target{i+1}' for i in range(NUM_TARGETS)]
columns = feature_columns + target_columns

# Create a DataFrame
df = pd.DataFrame(data, columns=columns)

# Save the DataFrame to CSV
df.to_csv('training_data.csv', index=False)

print("Synthetic data generated and saved to 'training_data.csv'")
