from neural_network import NeuralNetwork
from utils import normalize_data, plot_loss
import numpy as np

# Create a simple XOR problem
X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]]).T
y = np.array([[0], [1], [1], [0]])

# Initialize neural network with architecture [2, 3, 1]
nn = NeuralNetwork([2, 3, 1])

# Forward pass example
output = nn.forward(X)
print("Network output:", output[-1])
