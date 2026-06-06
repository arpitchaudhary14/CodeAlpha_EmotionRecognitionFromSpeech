import os
import numpy as np
import logging
import gc
import traceback
import soundfile as sf
import scipy.signal
import python_speech_features

def extract_features(file_path, max_pad_len=400, augment=False):
    """
    Extracts MFCC features safely using python_speech_features and soundfile,
    completely bypassing Librosa, Numba, and Resampy.
    """
    try:
        audio, sample_rate = sf.read(file_path)
        
        # Convert stereo to mono if necessary
        if len(audio.shape) > 1:
            audio = np.mean(audio, axis=1)

        # Resample to 22050 Hz using scipy (safe, no Numba) if needed
        TARGET_SR = 22050
        if sample_rate != TARGET_SR:
            num_samples = int(len(audio) * float(TARGET_SR) / sample_rate)
            audio = scipy.signal.resample(audio, num_samples)
            sample_rate = TARGET_SR
            
        # Truncate audio to max 10 seconds to save memory
        max_length = sample_rate * 10
        if len(audio) > max_length:
            audio = audio[:max_length]
            
        # To mimic librosa's default: hop_length=512, n_fft=2048 at 22050 Hz
        # winlen = 2048 / 22050 = ~0.0928 seconds
        # winstep = 512 / 22050 = ~0.0232 seconds
        mfccs_raw = python_speech_features.mfcc(
            audio,
            samplerate=sample_rate,
            winlen=0.0928,
            winstep=0.0232,
            numcep=40,
            nfilt=128,
            nfft=2048,
            appendEnergy=False
        )
        
        # python_speech_features returns (Time, NumCep). 
        # EmotionCNN expects (NumCep, Time) which is (40, T).
        mfccs = mfccs_raw.T
        
        # Explicit memory cleanup
        del audio
        del mfccs_raw
        gc.collect()
        
        current_len = mfccs.shape[1]
        
        if current_len > max_pad_len:
            mfccs = mfccs[:, :max_pad_len]
        elif current_len < max_pad_len:
            pad_width = max_pad_len - current_len
            mfccs = np.pad(mfccs, pad_width=((0, 0), (0, pad_width)), mode='constant')
            
        return mfccs

    except Exception as e:
        logging.error(traceback.format_exc())
        print(f"Error encountered while parsing file: {file_path}")
        print(f"Exception details: {e}")
        return None
