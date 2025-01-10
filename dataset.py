import numpy as np
import torch
import pandas as pd
from torch.utils.data import Dataset

class RISTrainingDataset(Dataset):
    """
    A PyTorch Dataset class for loading and processing RIS optimization data.
    """
    def __init__(self, data_file, transform=None):
        """
        Args:
            data_file (str): Path to the CSV file containing the dataset.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        # Load the dataset from a CSV file
        self.data = pd.read_csv(data_file)  # Ensure proper data is loaded
        
        # Convert all columns to numeric, coercing errors to NaN (optional: you can handle this differently)
        self.data = self.data.apply(pd.to_numeric, errors='coerce')
        
        # Drop rows with NaN values (if any)
        self.data = self.data.dropna()

        self.transform = transform

    def __len__(self):
        """Return the total number of samples in the dataset."""
        return len(self.data)

    def __getitem__(self, idx):
        """Retrieve a single sample from the dataset."""
        if torch.is_tensor(idx):
            idx = idx.tolist()

        # Ensure that features and target are numerical values (convert to float32)
        features = self.data.iloc[idx, :-64].values.astype(np.float32)  # Input features (first 128 columns)
        target = self.data.iloc[idx, -64:].values.astype(np.float32)  # Target values (last 64 columns)

        # Ensure correct tensor types
        features_tensor = torch.tensor(features, dtype=torch.float32)
        target_tensor = torch.tensor(target, dtype=torch.float32)

        print(f"Features data type: {features_tensor.dtype}, Target data type: {target_tensor.dtype}")  # Debugging line

        # Return as a dictionary
        sample = {'features': features_tensor, 'target': target_tensor}

        if self.transform:
            sample = self.transform(sample)

        return sample
