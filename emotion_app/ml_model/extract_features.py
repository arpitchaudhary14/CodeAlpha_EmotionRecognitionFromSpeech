import os
# Force Numba to stay fully disabled to prevent LLVM OOM crashes on Render.
os.environ["NUMBA_DISABLE_JIT"] = "1"

import librosa
import numpy as np
import logging
import gc
import traceback
import scipy.fftpack

def compute_mfcc_manual(y, sr, n_mfcc=40, n_fft=2048, hop_length=512, n_mels=128):
    """
    Manually computes MFCC to bypass librosa.feature.mfcc entirely.
    This guarantees zero calls to Numba-decorated utility functions,
    avoiding 'get_call_template' crashes when JIT is disabled.
    """
    # 1. Compute STFT (uses pure numpy)
    D = np.abs(librosa.stft(y, n_fft=n_fft, hop_length=hop_length))**2
    
    # 2. Apply Mel filterbank (uses pure numpy)
    mel_basis = librosa.filters.mel(sr=sr, n_fft=n_fft, n_mels=n_mels)
    S = np.dot(mel_basis, D)
    
    # 3. Convert to DB scale (uses pure numpy)
    S_db = librosa.power_to_db(S)
    
    # 4. Discrete Cosine Transform for MFCCs (uses pure scipy)
    mfccs = scipy.fftpack.dct(S_db, axis=0, type=2, norm='ortho')[:n_mfcc]
    return mfccs

def extract_features(file_path, max_pad_len=400, augment=False):
    """
    Extracts MFCC features safely without Numba or Resampy.
    """
    try:
        logging.warning("EF1 Before librosa.load")
        
        # res_type="scipy" forces librosa to avoid resampy (which uses Numba)
        audio, sample_rate = librosa.load(
            file_path,
            sr=22050,
            mono=True,
            duration=10.0,
            res_type="scipy"
        )
        logging.warning("AAA After librosa.load")
        
        # Augmentation is stripped because pitch_shift and time_stretch natively require Numba/Resampy
        if augment:
            logging.warning("Data augmentation skipped: natively requires Numba.")

        logging.warning("BBB Before MFCC")
        mfccs = compute_mfcc_manual(audio, sample_rate, n_mfcc=40)
        logging.warning("CCC After MFCC")
        
        del audio
        gc.collect()
        
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
        logging.error(traceback.format_exc())
        print(f"Error encountered while parsing file: {file_path}")
        print(f"Exception details: {e}")
        return None
