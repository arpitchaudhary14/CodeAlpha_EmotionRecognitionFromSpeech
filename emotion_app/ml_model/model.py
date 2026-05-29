import torch
import torch.nn as nn
import torch.nn.functional as F

class EmotionCNN(nn.Module):
    """
    A simple yet effective Convolutional Neural Network (CNN) for Speech Emotion Recognition.
    
    This network takes 2D MFCC spectrograms as input and outputs a classification
    across 4 emotion classes: Neutral, Happy, Sad, Angry.
    """
    def __init__(self, num_classes=4):
        super(EmotionCNN, self).__init__()
        
        # -----------------------------------------------------------
        # Convolutional Block 1
        # -----------------------------------------------------------
        # Input shape: (Batch, 1, height, max_pad_len)
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=32, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        # MaxPool reduces the spatial dimensions (height and width) by half
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        # Dropout helps prevent the model from memorizing the training data (overfitting)
        self.dropout1 = nn.Dropout(0.2)
        
        # -----------------------------------------------------------
        # Convolutional Block 2
        # -----------------------------------------------------------
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.dropout2 = nn.Dropout(0.2)
        
        # -----------------------------------------------------------
        # Convolutional Block 3
        # -----------------------------------------------------------
        self.conv3 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(128)
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.dropout3 = nn.Dropout(0.2)
        
        # -----------------------------------------------------------
        # Adaptive Pooling
        # -----------------------------------------------------------
        # AdaptiveAvgPool2d automatically calculates the correct pooling kernel size 
        # to squash the remaining height and width to exactly (2, 2). 
        # This is incredibly useful because it makes our Linear layer size constant (128 * 2 * 2), 
        # even if we change `max_pad_len` in the feature extractor later!
        self.adaptive_pool = nn.AdaptiveAvgPool2d((2, 2))
        
        # -----------------------------------------------------------
        # Fully Connected (Linear) Blocks
        # -----------------------------------------------------------
        # Flattened size = 128 (channels from conv3) * 2 (height) * 2 (width) = 512
        self.fc1 = nn.Linear(in_features=512, out_features=64)
        self.dropout4 = nn.Dropout(0.3)
        # Final output layer giving the raw scores (logits) for our 4 emotions
        self.fc2 = nn.Linear(in_features=64, out_features=num_classes)

    def forward(self, x):
        """
        Defines the forward pass (how data flows through the network).
        """
        # Pass through Block 1: Convolution -> BatchNorm -> ReLU Activation -> MaxPool -> Dropout
        x = self.pool1(F.relu(self.bn1(self.conv1(x))))
        x = self.dropout1(x)
        
        # Pass through Block 2
        x = self.pool2(F.relu(self.bn2(self.conv2(x))))
        x = self.dropout2(x)
        
        # Pass through Block 3
        x = self.pool3(F.relu(self.bn3(self.conv3(x))))
        x = self.dropout3(x)
        
        # Apply Adaptive Pooling to standardize the dimensions before the Linear layer
        x = self.adaptive_pool(x)
        
        # Flatten the tensor from (Batch, 128, 2, 2) to (Batch, 512)
        x = torch.flatten(x, start_dim=1)
        
        # Pass through Fully Connected layers
        x = F.relu(self.fc1(x))
        x = self.dropout4(x)
        
        # Final output (no ReLU here because these are the raw class scores)
        x = self.fc2(x)
        
        return x

# Quick test to verify the model architecture and shapes
if __name__ == "__main__":
    # Create a dummy batch of 2 MFCC samples: (batch_size=2, channels=1, n_mfcc=40, max_pad_len=400)
    dummy_input = torch.randn(2, 1, 40, 400)
    
    # Initialize model
    model = EmotionCNN(num_classes=4)
    
    # Forward pass
    output = model(dummy_input)
    
    print("Model initialized successfully!")
    print(f"Input shape: {dummy_input.shape}")
    print(f"Output shape: {output.shape} (Batch Size, Num Classes)")
