# app.py

import streamlit as st
import numpy as np
from neural_network import initialize_parameters, forward_propagation, backward_propagation, update_parameters, compute_cost

st.title('Two-Layer Neural Network Demonstration')

# Input parameters
input_dim = st.number_input('Number of Input Features:', min_value=1, value=3)
hidden_dim = st.number_input('Number of Hidden Neurons:', min_value=1, value=4)
output_dim = st.number_input('Number of Output Neurons:', min_value=1, value=1)
learning_rate = st.number_input('Learning Rate:', min_value=0.0001, value=0.01, format="%.4f")
num_iterations = st.number_input('Number of Iterations:', min_value=1, value=1000)

# Sample input data
st.subheader('Input Data')
X = np.random.randn(input_dim, 1)
Y = np.random.randint(0, 2, (output_dim, 1))
st.write('Sample Input (X):', X)
st.write('Sample True Output (Y):', Y)

# Initialize parameters
parameters = initialize_parameters(input_dim, hidden_dim, output_dim)

# Display initialized weights and biases
st.subheader('Initialized Parameters')
st.write('Weight Matrix W1:', parameters['W1'])
st.write('Bias Vector b1:', parameters['b1'])
st.write('Weight Matrix W2:', parameters['W2'])
st.write('Bias Vector b2:', parameters['b2'])

# Training the neural network
costs = []
for i in range(num_iterations):
    # Forward propagation
    A2, cache = forward_propagation(X, parameters)
    # Compute cost
    cost = compute_cost(A2, Y)
    costs.append(cost)
    # Backward propagation
    grads = backward_propagation(parameters, cache, X, Y)
    # Update parameters
    parameters = update_parameters(parameters, grads, learning_rate)

# Display final weights and biases
::contentReference[oaicite:0]{index=0}
 
