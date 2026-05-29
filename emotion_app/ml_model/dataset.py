import os
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split

# Import the feature extraction function we created
from extract_features import extract_features

# RAVDESS Emotion mapping based on filename conventions.
# The RAVDESS filename format is: Modality-VocalChannel-Emotion-EmotionalIntensity-Statement-Repetition-Actor.wav
# Example: 03-01-01-01-01-01-01.wav -> The 3rd part '01' is the emotion.
# 01 = neutral, 02 = calm, 03 = happy, 04 = sad, 05 = angry, 06 = fearful, 07 = disgust, 08 = surprised
# We filter to only include the requested emotions and encode them numerically.
EMOTION_MAP = {
    '03': 0, # happy
    '04': 1, # sad
    '05': 2, # angry
    '01': 3  # neutral
}

class RAVDESSDataset(Dataset):
    """
    Custom PyTorch Dataset for RAVDESS Speech Emotion data.
    """
    def __init__(self, file_paths, labels, max_pad_len=400):
        """
        Args:
            file_paths (list): List of paths to the .wav files.
            labels (list): Corresponding numeric labels.
            max_pad_len (int): Length to pad/truncate MFCCs to.
        """
        self.file_paths = file_paths
        self.labels = labels
        self.max_pad_len = max_pad_len

    def __len__(self):
        return len(self.file_paths)

    def __getitem__(self, idx):
        file_path = self.file_paths[idx]
        label = self.labels[idx]

        # 1. Extract MFCC features using our custom function
        features = extract_features(file_path, max_pad_len=self.max_pad_len)
        
        # Fallback in case a file is corrupted or cannot be read
        if features is None:
            features = np.zeros((40, self.max_pad_len))

        # 2. Convert to PyTorch tensors
        # We add a channel dimension using unsqueeze(0) because CNNs 
        # usually expect input shapes like (Batch, Channels, Height, Width).
        # Our shape becomes (1, 40, max_pad_len)
        feature_tensor = torch.tensor(features, dtype=torch.float32).unsqueeze(0)
        label_tensor = torch.tensor(label, dtype=torch.long)

        return feature_tensor, label_tensor


def prepare_dataset(dataset_dir, test_size=0.2, random_state=42):
    """
    Parses the dataset directory, filters requested emotions, 
    and splits the data into train and test sets.

    Args:
        dataset_dir (str): Root directory containing RAVDESS Actor_* folders.
        test_size (float): Proportion of dataset to include in the test split.
        random_state (int): Random seed for reproducibility.

    Returns:
        train_paths, test_paths, train_labels, test_labels
    """
    valid_paths = []
    valid_labels = []

    # Iterate through the Actor directories
    for root, _, files in os.walk(dataset_dir):
        for file in files:
            if file.endswith('.wav'):
                # Split filename to extract the emotion code
                parts = file.split('-')
                
                # Ensure the filename follows the RAVDESS convention
                if len(parts) == 7:
                    emotion_code = parts[2]
                    
                    # Filter only the emotions we want
                    if emotion_code in EMOTION_MAP:
                        file_path = os.path.join(root, file)
                        label = EMOTION_MAP[emotion_code]
                        
                        valid_paths.append(file_path)
                        valid_labels.append(label)

    print(f"Found {len(valid_paths)} valid audio files matching requested emotions.")

    if len(valid_paths) == 0:
         raise ValueError("No matching audio files found. Please check your dataset directory.")

    # Split the dataset into training and testing sets
    train_paths, test_paths, train_labels, test_labels = train_test_split(
        valid_paths, 
        valid_labels, 
        test_size=test_size, 
        random_state=random_state,
        stratify=valid_labels # Stratify to maintain class distribution in splits
    )

    return train_paths, test_paths, train_labels, test_labels


def get_dataloaders(dataset_dir, batch_size=32, test_size=0.2):
    """
    Utility function to quickly get PyTorch DataLoaders for training and testing.
    """
    train_paths, test_paths, train_labels, test_labels = prepare_dataset(
        dataset_dir, test_size=test_size
    )

    # Initialize PyTorch datasets
    train_dataset = RAVDESSDataset(train_paths, train_labels)
    test_dataset = RAVDESSDataset(test_paths, test_labels)

    # Create PyTorch dataloaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, test_loader
