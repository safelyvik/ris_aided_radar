import torch
import numpy as np
from model import RISOptimizationModel

# Configuration
NUM_RIS_ELEMENTS = 64
INPUT_DIM = 128  # Example input dimension
HIDDEN_DIM = 256
OUTPUT_DIM = NUM_RIS_ELEMENTS
MODEL_PATH = "ris_optimization_model.pth"

# Load the trained model
def load_model():
    """
    Loads the trained RIS optimization model.
    """
    model = RISOptimizationModel(NUM_RIS_ELEMENTS, INPUT_DIM, HIDDEN_DIM, OUTPUT_DIM)
    model.load_state_dict(torch.load(MODEL_PATH))
    model.eval()  # Set the model to evaluation mode
    return model

# Predict function
def predict_phase_shifts(input_data):
    """
    Predicts the phase shift matrix for given input data.
    
    Args:
        input_data (numpy array): Input data of shape (N, INPUT_DIM), where N is the number of samples.

    Returns:
        numpy array: Predicted phase shift matrix of shape (N, NUM_RIS_ELEMENTS).
    """
    model = load_model()
    input_tensor = torch.tensor(input_data, dtype=torch.float32)
    with torch.no_grad():
        output_tensor = model(input_tensor)
    return output_tensor.numpy()

# Example usage
if __name__ == "__main__":
    # Example input data (replace with real test data)
    example_input = np.random.rand(1, INPUT_DIM)  # 1 sample with INPUT_DIM features
    
    # Predict phase shifts
    phase_shifts = predict_phase_shifts(example_input)
    
    # Display the predicted phase shift matrix
    print("Predicted Phase Shift Matrix:")
    print(phase_shifts)
