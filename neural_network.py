# neural_network.py

import numpy as np

def sigmoid(x):
    """Sigmoid activation function."""
    return 1 / (1 + np.exp(-x))

def sigmoid_derivative(x):
    """Derivative of the sigmoid function."""
    return sigmoid(x) * (1 - sigmoid(x))

def initialize_parameters(input_dim, hidden_dim, output_dim):
    """
    Initialize weights and biases for a two-layer neural network.
    Args:
        input_dim (int): Number of input features.
        hidden_dim (int): Number of neurons in the hidden layer.
        output_dim (int): Number of output neurons.
    Returns:
        dict: Dictionary containing initialized weights and biases.
    """
    np.random.seed(42)  # For reproducibility
    parameters = {
        'W1': np.random.randn(hidden_dim, input_dim) * 0.01,
        'b1': np.zeros((hidden_dim, 1)),
        'W2': np.random.randn(output_dim, hidden_dim) * 0.01,
        'b2': np.zeros((output_dim, 1))
    }
    return parameters

def forward_propagation(X, parameters):
    """
    Perform forward propagation through the network.
    Args:
        X (ndarray): Input data of shape (input_dim, number_of_examples).
        parameters (dict): Dictionary containing weights and biases.
    Returns:
        tuple: Tuple containing activations and intermediate values.
    """
    W1 = parameters['W1']
    b1 = parameters['b1']
    W2 = parameters['W2']
    b2 = parameters['b2']

    Z1 = np.dot(W1, X) + b1
    A1 = sigmoid(Z1)
    Z2 = np.dot(W2, A1) + b2
    A2 = sigmoid(Z2)

    cache = {'Z1': Z1, 'A1': A1, 'Z2': Z2, 'A2': A2}
    return A2, cache

def compute_cost(A2, Y):
    """
    Compute the cross-entropy cost.
    Args:
        A2 (ndarray): The output of the neural network (predictions).
        Y (ndarray): True labels.
    Returns:
        float: Cross-entropy cost.
    """
    m = Y.shape[1]
    cost = -np.sum(Y * np.log(A2) + (1 - Y) * np.log(1 - A2)) / m
    return np.squeeze(cost)

def backward_propagation(parameters, cache, X, Y):
    """
    Perform backward propagation to compute gradients.
    Args:
        parameters (dict): Dictionary containing weights and biases.
        cache (dict): Dictionary containing intermediate values from forward propagation.
        X (ndarray): Input data.
        Y (ndarray): True labels.
    Returns:
        dict: Dictionary containing gradients of weights and biases.
    """
    m = X.shape[1]

    W2 = parameters['W2']
    A1 = cache['A1']
    A2 = cache['A2']

    dZ2 = A2 - Y
    dW2 = np.dot(dZ2, A1.T) / m
    db2 = np.sum(dZ2, axis=1, keepdims=True) / m

    dZ1 = np.dot(W2.T, dZ2) * sigmoid_derivative(cache['Z1'])
    dW1 = np.dot(dZ1, X.T) / m
    db1 = np.sum(dZ1, axis=1, keepdims=True) / m

    grads = {'dW1': dW1, 'db1': db1, 'dW2': dW2, 'db2': db2}
    return grads

def update_parameters(parameters, grads, learning_rate=0.01):
    """
    Update parameters using gradient descent.
    Args:
        parameters (dict): Dictionary containing weights and biases.
        grads (dict): Dictionary containing gradients of weights and biases.
        learning_rate (float): Learning rate for gradient descent.
    Returns:
        dict: Updated parameters.
    """
    parameters['W1'] -= learning_rate * grads['dW1']
    parameters['b1'] -= learning_rate * grads['db1']
    parameters['W2'] -= learning_rate * grads['dW2']
    parameters['b2'] -= learning_rate * grads['db2']
    return parameters
