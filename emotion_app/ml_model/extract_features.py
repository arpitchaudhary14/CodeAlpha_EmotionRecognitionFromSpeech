import os
os.environ["NUMBA_DISABLE_JIT"] = "1"

import librosa
import numpy as np
import logging
import gc

def add_noise(data, noise_factor=0.005):
    """Adds random Gaussian noise to the audio signal."""
    noise = np.random.randn(len(data))
    augmented_data = data + noise_factor * noise
    return augmented_data

def shift_pitch(data, sample_rate, n_steps=2):
    """Shifts the pitch of the audio signal."""
    return librosa.effects.pitch_shift(y=data, sr=sample_rate, n_steps=n_steps)

def stretch_time(data, rate=1.2):
    """Stretches or compresses the time of the audio signal."""
    return librosa.effects.time_stretch(y=data, rate=rate)

def extract_features(file_path, max_pad_len=400, augment=False):
    """
    Extracts MFCC features from an audio file, with optional data augmentation.
    
    Args:
        file_path (str): Path to the .wav audio file.
        max_pad_len (int, optional): Fixed length to pad or truncate MFCCs to. Defaults to 400.
        augment (bool, optional): Whether to apply random audio augmentations. Defaults to False.

    Returns:
        np.ndarray: A 2D NumPy array of shape (n_mfcc, max_pad_len).
    """
    try:
        # 1. Load the audio file (limit to max 10.0 seconds for memory safety on Render)
        logging.warning("EF1 Before librosa.load")
        audio, sample_rate = librosa.load(
            file_path,
            sr=22050,
            mono=True,
            duration=10.0
        )
        logging.warning("EF2 After librosa.load")
        
        # 2. Data Augmentation (Randomly applied during training)
        if augment:
            # Randomly decide which augmentations to apply (each has a 50% chance)
            if np.random.rand() < 0.5:
                # Add slight background noise (kept small between 0.001 to 0.005)
                # Avoids corrupting emotion semantics with extreme noise
                noise_factor = np.random.uniform(0.001, 0.005)
                audio = add_noise(audio, noise_factor)
                
            if np.random.rand() < 0.5:
                # Slight pitch shifting between -2 and +2 semitones
                # Extreme pitch shifts alter emotion characteristics, so keep it tight
                n_steps = np.random.uniform(-2, 2)
                audio = shift_pitch(audio, sample_rate, n_steps)
                
            if np.random.rand() < 0.5:
                # Slight time stretching (0.8x to 1.2x speed)
                # Avoids aggressive speed changes that sound robotic
                rate = np.random.uniform(0.8, 1.2)
                audio = stretch_time(audio, rate)
        
        # 3. Extract Features
        # Extract standard MFCCs
        logging.warning("EF3 Before MFCC")
        mfccs = librosa.feature.mfcc(y=audio, sr=sample_rate, n_mfcc=40)
        logging.warning("EF4 After MFCC")
        
        # Explicit memory cleanup for the raw audio array before padding operations
        del audio
        gc.collect()
        
        # 4. Pad or truncate to ensure a fixed size output along the time axis
        logging.warning("EF5 Before Padding")
        current_len = mfccs.shape[1]
        
        if current_len > max_pad_len:
            mfccs = mfccs[:, :max_pad_len]
        elif current_len < max_pad_len:
            pad_width = max_pad_len - current_len
            mfccs = np.pad(mfccs, pad_width=((0, 0), (0, pad_width)), mode='constant')
            
        logging.warning("EF6 Returning Features")
        return mfccs

    except Exception as e:
        print(f"Error encountered while parsing file: {file_path}")
        print(f"Exception details: {e}")
        return None
