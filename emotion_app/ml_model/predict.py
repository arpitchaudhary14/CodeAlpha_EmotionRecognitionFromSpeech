import os
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
    
    # 1. Resolve absolute model path
    # We use os.path.dirname(__file__) to ensure it looks in the same directory as this script
    abs_model_path = os.path.join(os.path.dirname(__file__), model_path)
    
    if not os.path.exists(abs_model_path):
        return {
            "error": f"Model file not found at {abs_model_path}. Please train the model first."
        }

    # 2. Extract Features
    # The feature extractor will pad/truncate to guarantee shape (40, max_pad_len)
    features = extract_features(file_path, max_pad_len=max_pad_len)
    
    if features is None:
        return {
            "error": "Failed to extract features from the audio file."
        }
        
    # 3. Prepare Tensor Dimensions
    # PyTorch CNNs expect a 4D tensor input: (Batch_Size, Channels, Height, Width)
    # Our features array is 2D: (40, max_pad_len).
    # We use unsqueeze(0) twice: 
    # First to add the Channel dimension -> (1, 40, max_pad_len)
    # Second to add the Batch dimension -> (1, 1, 40, max_pad_len)
    feature_tensor = torch.tensor(features, dtype=torch.float32).unsqueeze(0).unsqueeze(0)

    # 4. Hardware Setup (Fallback to CPU for inference if GPU is busy/unavailable)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    feature_tensor = feature_tensor.to(device)

    # 5. Initialize Model and Load Weights
    model = EmotionCNN(num_classes=4).to(device)
    
    # Load the state_dict. map_location ensures it loads on CPU safely if it was trained on GPU
    model.load_state_dict(torch.load(abs_model_path, map_location=device, weights_only=True))
    
    # IMPORTANT: Set model to evaluation mode! 
    # This disables Dropout and BatchNorm, ensuring deterministic predictions.
    model.eval()

    # 6. Perform Inference
    # torch.no_grad() disables gradient calculation, saving memory and speeding up prediction
    with torch.no_grad():
        # Get raw, unnormalized scores (logits) from the network
        outputs = model(feature_tensor)
        
        # Apply Softmax to convert raw scores into probabilities that sum to 1.0
        # dim=1 means we apply softmax across the class dimension
        probabilities = F.softmax(outputs, dim=1)
        
        # Get the highest probability (confidence) and its corresponding index
        confidence, predicted_idx = torch.max(probabilities, 1)

    # Extract the standard Python values from the PyTorch tensors using .item()
    predicted_label = REVERSE_EMOTION_MAP.get(predicted_idx.item(), "Unknown")
    confidence_score = confidence.item()

    # Extract all probabilities and map them
    # probabilities[0] because of the batch dimension
    probs_list = probabilities[0].tolist()
    emotion_probs = {}
    for idx, prob in enumerate(probs_list):
        emotion_name = REVERSE_EMOTION_MAP.get(idx, "Unknown")
        emotion_probs[emotion_name] = round(prob, 4)

    # 7. Return Result
    return {
        "emotion": predicted_label,
        "confidence": round(confidence_score, 4),
        "probabilities": emotion_probs
    }

if __name__ == "__main__":
    # Example usage for testing the script directly:
    test_audio = "path/to/test_audio.wav"
    if os.path.exists(test_audio):
        result = predict_emotion(test_audio)
        print(f"Prediction Result: {result}")
    else:
        print("Please provide a valid audio file path to test.")
