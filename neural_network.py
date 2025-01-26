# neural_network.py

import numpy as np

def sigmoid(x):
    """Sigmoid activation function."""
    return 1 / (1 + np.exp(-x))

def sigmoid_derivative(x):
    """Derivative of the sigmoid function."""
    return sigmoid(x) * (1 - sigmoid(x))

def forward_propagation(x, w, b):
    """
    Perform forward propagation.
    Args:
        x (float): Input value.
        w (float): Weight.
        b (float): Bias.
    Returns:
        output (float): Output after activation.
        net_input (float): Weighted sum before activation.
    """
    net_input = w * x + b
    output = sigmoid(net_input)
    return output, net_input

def backward_propagation(x, y, output, net_input):
    """
    Perform backward propagation to compute gradients.
    Args:
        x (float): Input value.
        y (float): True output value.
        output (float): Predicted output.
        net_input (float): Weighted sum before activation.
    Returns:
        d_loss_d_w (float): Gradient of loss with respect to weight.
        d_loss_d_b (float): Gradient of loss with respect to bias.
        error (float): Difference between predicted and true output.
    """
    error = output - y
    d_output_d_net = sigmoid_derivative(net_input)
    d_loss_d_w = error * d_output_d_net * x
    d_loss_d_b = error * d_output_d_net
    return d_loss_d_w, d_loss_d_b, error
