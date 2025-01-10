import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from dataset import RISTrainingDataset
from model import RISOptimizationModel

# Set random seed for reproducibility
torch.manual_seed(42)

# Configuration
NUM_RIS_ELEMENTS = 64
INPUT_DIM = 128  # Example input dimension
HIDDEN_DIM = 256
OUTPUT_DIM = NUM_RIS_ELEMENTS
LEARNING_RATE = 0.001
NUM_EPOCHS = 1000
BATCH_SIZE = 16

# Dataset and DataLoader
train_dataset = RISTrainingDataset("training_data.csv")
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)

# Model, Loss, Optimizer
model = RISOptimizationModel(NUM_RIS_ELEMENTS, INPUT_DIM, HIDDEN_DIM, OUTPUT_DIM)
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

# Training Loop
def train_model():
    """
    Trains the RIS optimization model.
    """
    model.train()  # Set the model to training mode
    for epoch in range(NUM_EPOCHS):
        epoch_loss = 0.0
        for batch_idx, sample in enumerate(train_loader):
            # Extract inputs and targets
            inputs = sample['features']
            targets = sample['target']

            # Debugging: Check the types of inputs and targets
            print(f"Inputs data type: {inputs.dtype}, Targets data type: {targets.dtype}")

            # Convert inputs and targets to float if they are not already
            if inputs.dtype != torch.float32:
                inputs = inputs.float()
            if targets.dtype != torch.float32:
                targets = targets.float()

            # Forward pass
            optimizer.zero_grad()
            outputs = model(inputs)

            # Loss calculation
            loss = criterion(outputs, targets)
            epoch_loss += loss.item()

            # Backward pass and optimization
            loss.backward()
            optimizer.step()

        print(f"Epoch {epoch + 1}/{NUM_EPOCHS}, Loss: {epoch_loss / len(train_loader):.4f}")

    print("Training completed.")
    torch.save(model.state_dict(), "ris_optimization_model.pth")
    print("Model saved as 'ris_optimization_model.pth'.")

if __name__ == "__main__":
    train_model()
