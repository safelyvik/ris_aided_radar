import torch
import torch.nn as nn

class RISOptimizationModel(nn.Module):
    """
    Neural Network Model for RIS Phase Optimization.
    """
    def __init__(self, num_ris_elements, input_dim, hidden_dim, output_dim):
        super(RISOptimizationModel, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, output_dim)
        self.activation = nn.ReLU()
        self.output_activation = nn.Tanh()  # Phase shift values are normalized

    def forward(self, x):
        """
        Forward pass for the model.
        :param x: Input tensor.
        :return: Optimized phase shift matrix.
        """
        x = self.activation(self.fc1(x))
        x = self.activation(self.fc2(x))
        x = self.output_activation(self.fc3(x))  # Normalized to [-1, 1]
        return x
