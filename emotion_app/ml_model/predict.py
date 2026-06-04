import os
import gc
import logging
import torch
import torch.nn.functional as F

# Import our custom feature extraction and model architecture
from .extract_features import extract_features
from .model import EmotionCNN

# Reverse mapping to convert numeric predictions back to human-readable labels
# (Matches the encoding from dataset.py: 0=Happy, 1=Sad, 2=Angry)
REVERSE_EMOTION_MAP = {
    0: 'Happy',
    1: 'Sad',
    2: 'Angry',
    3: 'Neutral'
}

# --- GLOBAL MODEL CACHING ---
# To prevent Out-Of-Memory (OOM) errors on Render, we load the model into memory exactly once 
# per worker process, rather than reloading it on every single request.
_GLOBAL_MODEL = None
_GLOBAL_DEVICE = None

def get_model(model_path='best_model.pth'):
    """
    Returns the loaded PyTorch model and device.
    Initializes the model only on the first call (Singleton pattern).
    """
    global _GLOBAL_MODEL, _GLOBAL_DEVICE
    
    if _GLOBAL_MODEL is None:
        _GLOBAL_DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        abs_model_path = os.path.join(os.path.dirname(__file__), model_path)
        
        if not os.path.exists(abs_model_path):
            raise FileNotFoundError(f"Model file not found at {abs_model_path}. Please train the model first.")
            
        # Initialize model architecture and move to device
        model = EmotionCNN(num_classes=4).to(_GLOBAL_DEVICE)
        
        # Load weights
        model.load_state_dict(torch.load(abs_model_path, map_location=_GLOBAL_DEVICE, weights_only=True))
        
        # Set to evaluation mode for deterministic predictions
        model.eval()
        
        _GLOBAL_MODEL = model
        
    return _GLOBAL_MODEL, _GLOBAL_DEVICE


def predict_emotion(file_path, model_path='best_model.pth', max_pad_len=400):
    """
    Predicts the emotion of a given audio file using the trained PyTorch model.
    
    Args:
        file_path (str): Path to the .wav audio file to be analyzed.
        model_path (str): Path to the saved model weights (.pth file).
        max_pad_len (int): The padding length used during training.
        
    Returns:
        dict: A dictionary containing the predicted 'emotion' (str) and 
              the 'confidence' score (float between 0 and 1), or an error 
              message if processing fails.
    """
    try:
        logging.warning("STEP 1: Prediction Started")
        
        # 1. Get Global Model (Loads only once per process)
        try:
            model, device = get_model(model_path)
        except Exception as e:
            return {"error": str(e)}

        # 2. Extract Features
        # The feature extractor will pad/truncate to guarantee shape (40, max_pad_len)
        features = extract_features(file_path, max_pad_len=max_pad_len)
        logging.warning("STEP 2: Features Extracted")
        
        if features is None:
            return {
                "error": "Failed to extract features from the audio file."
            }
            
        # 3. Prepare Tensor Dimensions
        feature_tensor = torch.tensor(features, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
        feature_tensor = feature_tensor.to(device)
        logging.warning("STEP 3: Tensor Created")

        # 4. Perform Inference
        # torch.no_grad() disables gradient calculation, saving memory and speeding up prediction
        with torch.no_grad():
            outputs = model(feature_tensor)
            probabilities = F.softmax(outputs, dim=1)
            confidence, predicted_idx = torch.max(probabilities, 1)
            
        logging.warning("STEP 4: Inference Complete")

        # Extract the standard Python values from the PyTorch tensors using .item()
        predicted_label = REVERSE_EMOTION_MAP.get(predicted_idx.item(), "Unknown")
        confidence_score = confidence.item()

        # Extract all probabilities and map them
        probs_list = probabilities[0].tolist()
        emotion_probs = {}
        for idx, prob in enumerate(probs_list):
            emotion_name = REVERSE_EMOTION_MAP.get(idx, "Unknown")
            emotion_probs[emotion_name] = round(prob, 4)

        # 5. Clean up local tensors to eagerly free memory before garbage collection
        del feature_tensor
        del outputs
        del probabilities

        # 6. Return Result
        logging.warning("STEP 5: Returning JSON")
        return {
            "emotion": predicted_label,
            "confidence": round(confidence_score, 4),
            "probabilities": emotion_probs
        }
    finally:
        # Force garbage collection to recover RAM on Render's constrained instances
        gc.collect()

if __name__ == "__main__":
    # Example usage for testing the script directly:
    test_audio = "path/to/test_audio.wav"
    if os.path.exists(test_audio):
        result = predict_emotion(test_audio)
        print(f"Prediction Result: {result}")
    else:
        print("Please provide a valid audio file path to test.")

