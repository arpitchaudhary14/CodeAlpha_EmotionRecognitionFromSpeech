import os
import torch
import torch.nn as nn
import torch.optim as optim
from collections import Counter
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

# Import our custom dataset loader and model
from dataset import get_dataloaders
from model import EmotionCNN

# REVERSE_EMOTION_MAP for mapping numeric indices back to string labels for reports
REVERSE_EMOTION_MAP = {
    0: 'Happy',
    1: 'Sad',
    2: 'Angry',
    3: 'Neutral'
}

def train_model(dataset_dir, num_epochs=80, batch_size=32, learning_rate=0.001, model_save_path='best_model.pth'):
    """
    Trains the EmotionCNN model and evaluates it using a validation loop.
    
    Args:
        dataset_dir (str): Path to the directory containing the dataset.
        num_epochs (int): Number of times to iterate over the entire training dataset.
        batch_size (int): Number of samples processed before updating model weights.
        learning_rate (float): Step size for the optimizer.
        model_save_path (str): File name/path where the trained weights will be saved.
    """
    
    # 1. Hardware setup (Use GPU if available, otherwise fallback to CPU)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # 2. Load the Data
    print("Preparing dataloaders...")
    try:
        train_loader, val_loader = get_dataloaders(dataset_dir, batch_size=batch_size, test_size=0.2)
        
        # --- Class Distribution Analysis ---
        print("\n--- Class Distribution ---")
        train_counts = Counter(train_loader.dataset.labels)
        
        # Calculate class weights for CrossEntropyLoss to handle imbalance
        # Weight formula: Total Samples / (Number of Classes * Class Count)
        total_samples = len(train_loader.dataset.labels)
        num_classes = len(REVERSE_EMOTION_MAP)
        class_weights = [0.0] * num_classes
        
        for label_idx, count in train_counts.items():
            emotion_name = REVERSE_EMOTION_MAP.get(label_idx, f"Unknown ({label_idx})")
            print(f"{emotion_name}: {count} samples")
            
            # Prevent division by zero just in case
            if count > 0:
                class_weights[label_idx] = total_samples / (num_classes * count)
                
        print("--------------------------\n")
        
    except Exception as e:
        print(f"Failed to load dataset from {dataset_dir}. Error: {e}")
        return

    # 3. Initialize the Model, Loss Function, and Optimizer
    print("Initializing model...")
    # FUTURE UPGRADE: Placeholder for future CNN + BiLSTM architecture
    # model = EmotionCNN_BiLSTM(num_classes=4).to(device)
    model = EmotionCNN(num_classes=4).to(device)
    
    # Apply class weights to the loss function to penalize majority classes
    weights_tensor = torch.tensor(class_weights, dtype=torch.float32).to(device)
    criterion = nn.CrossEntropyLoss(weight=weights_tensor)
    
    # Adam is a widely used, robust optimizer
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    print(f"Starting training for {num_epochs} epochs...\n")

    best_val_acc = 0.0
    patience_counter = 0
    early_stopping_patience = 20

    # 4. The Training Loop
    for epoch in range(num_epochs):
        # --- TRAINING PHASE ---
        model.train() # Set the model to training mode (enables Dropout, etc.)
        running_train_loss = 0.0
        correct_train = 0
        total_train = 0

        for inputs, labels in train_loader:
            # Move data to the active device (GPU/CPU)
            inputs = inputs.to(device)
            labels = labels.to(device)

            # Zero the parameter gradients to prevent accumulation from previous iterations
            optimizer.zero_grad()

            # Forward pass: predict the emotions
            outputs = model(inputs)
            
            # Calculate the loss
            loss = criterion(outputs, labels)
            
            # Backward pass: calculate the gradients
            loss.backward()
            
            # Update weights
            optimizer.step()

            # Track statistics
            running_train_loss += loss.item() * inputs.size(0)
            
            # Get the index of the highest score (the predicted class)
            _, predicted = torch.max(outputs.data, 1)
            total_train += labels.size(0)
            correct_train += (predicted == labels).sum().item()

        # Calculate average training loss and accuracy for the epoch
        epoch_train_loss = running_train_loss / total_train
        epoch_train_acc = (correct_train / total_train) * 100

        # --- VALIDATION PHASE ---
        model.eval() # Set the model to evaluation mode (disables Dropout)
        running_val_loss = 0.0
        correct_val = 0
        total_val = 0

        # torch.no_grad() tells PyTorch we don't need to track gradients, saving memory and speeding up computation
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs = inputs.to(device)
                labels = labels.to(device)

                # Forward pass
                outputs = model(inputs)
                
                # Calculate loss
                loss = criterion(outputs, labels)
                
                # Track statistics
                running_val_loss += loss.item() * inputs.size(0)
                _, predicted = torch.max(outputs.data, 1)
                total_val += labels.size(0)
                correct_val += (predicted == labels).sum().item()

        # Calculate average validation loss and accuracy for the epoch
        epoch_val_loss = running_val_loss / total_val
        epoch_val_acc = (correct_val / total_val) * 100

        # 5. Print progress
        print(f"Epoch [{epoch+1}/{num_epochs}] "
              f"Train Loss: {epoch_train_loss:.4f}, Train Acc: {epoch_train_acc:.2f}% | "
              f"Val Loss: {epoch_val_loss:.4f}, Val Acc: {epoch_val_acc:.2f}%")
              
        # --- Model Checkpointing ---
        if epoch_val_acc > best_val_acc:
            best_val_acc = epoch_val_acc
            patience_counter = 0
            
            save_path = os.path.join(os.path.dirname(__file__), model_save_path)
            torch.save(model.state_dict(), save_path)
            print(f"--> New Best Model Saved! Validation Accuracy: {best_val_acc:.2f}%")
        else:
            patience_counter += 1
            
        # --- Early Stopping ---
        # Do not allow early stopping before epoch 30 as requested
        if epoch >= 30 and patience_counter >= early_stopping_patience:
            print(f"\nEarly stopping triggered after {epoch+1} epochs due to no improvement in validation accuracy.")
            break

    # 6. Advanced Evaluation Metrics
    print("\nTraining complete! Generating evaluation reports on best model...")
    
    # Load the best weights before evaluating
    save_path = os.path.join(os.path.dirname(__file__), model_save_path)
    if os.path.exists(save_path):
        model.load_state_dict(torch.load(save_path))
    
    model.eval()
    all_preds = []
    all_targets = []
    
    with torch.no_grad():
        for inputs, labels in val_loader:
            inputs = inputs.to(device)
            labels = labels.to(device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)
            
            all_preds.extend(predicted.cpu().numpy())
            all_targets.extend(labels.cpu().numpy())
            
    # Map back to string labels for sklearn report
    target_names = [REVERSE_EMOTION_MAP[i] for i in sorted(REVERSE_EMOTION_MAP.keys())]
    
    print("\n--- Classification Report ---")
    print(classification_report(all_targets, all_preds, target_names=target_names))
    
    # Generate Confusion Matrix
    cm = confusion_matrix(all_targets, all_preds)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=target_names, yticklabels=target_names)
    plt.title('Confusion Matrix - Emotion Validation')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    
    cm_path = os.path.join(os.path.dirname(__file__), 'confusion_matrix.png')
    plt.savefig(cm_path)
    print(f"\nConfusion matrix saved as: {cm_path}")
    print(f"Best Validation Accuracy: {best_val_acc:.2f}%")


if __name__ == "__main__":
    # Example usage:
    # Set this to the folder containing your RAVDESS dataset (e.g., the folder containing Actor_01, Actor_02, etc.)
    DATASET_DIRECTORY = "C:/Users/hp/django/EmotionSenseAI/dataset/RAVDESS" 
    
    # You can customize hyperparameters here
    train_model(
        dataset_dir=DATASET_DIRECTORY, 
        num_epochs=80, 
        batch_size=32, 
        learning_rate=0.0005,
        model_save_path='best_model.pth'
    )