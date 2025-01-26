# app.py

import streamlit as st
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
from neural_network import initialize_parameters, forward_propagation, backward_propagation, update_parameters, compute_cost

st.title('Two-Layer Neural Network Demonstration')

# Add network architecture visualization
def plot_network_architecture(input_dim, hidden_dim, output_dim):
    fig = go.Figure()
    
    # Add nodes
    layer_sizes = [input_dim, hidden_dim, output_dim]
    layer_names = ['Input Layer', 'Hidden Layer', 'Output Layer']
    colors = ['lightblue', 'lightgreen', 'salmon']
    
    for i, (size, name, color) in enumerate(zip(layer_sizes, layer_names, colors)):
        x = [i] * size
        y = np.linspace(-size/2, size/2, size)
        fig.add_trace(go.Scatter(
            x=x, y=y,
            mode='markers+text',
            marker=dict(size=20, color=color),
            text=[f'{name}<br>Node {j+1}' for j in range(size)],
            name=name
        ))
        
        # Add connections to next layer
        if i < len(layer_sizes) - 1:
            for y1 in np.linspace(-size/2, size/2, size):
                for y2 in np.linspace(-layer_sizes[i+1]/2, layer_sizes[i+1]/2, layer_sizes[i+1]):
                    fig.add_trace(go.Scatter(
                        x=[i, i+1],
                        y=[y1, y2],
                        mode='lines',
                        line=dict(color='gray', width=0.5),
                        showlegend=False
                    ))
    
    fig.update_layout(
        title='Neural Network Architecture',
        showlegend=True,
        height=400
    )
    return fig

# Input parameters
input_dim = st.number_input('Number of Input Features:', min_value=1, value=3)
hidden_dim = st.number_input('Number of Hidden Neurons:', min_value=1, value=4)
output_dim = st.number_input('Number of Output Neurons:', min_value=1, value=1)
learning_rate = st.number_input('Learning Rate:', min_value=0.0001, value=0.01, format="%.4f")
num_iterations = st.number_input('Number of Iterations:', min_value=1, value=1)

st.subheader('Network Architecture')
arch_fig = plot_network_architecture(input_dim, hidden_dim, output_dim)
st.plotly_chart(arch_fig)

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
predictions = []
iterations = []

for i in range(num_iterations):
    # Forward propagation
    A2, cache = forward_propagation(X, parameters)
    # Compute cost
    cost = compute_cost(A2, Y)
    costs.append(cost)
    predictions.append(A2[0][0])
    iterations.append(i)
    # Backward propagation
    grads = backward_propagation(parameters, cache, X, Y)
    # Update parameters
    parameters = update_parameters(parameters, grads, learning_rate)

# Enhanced training visualization
st.subheader('Training Progress')
col1, col2 = st.columns(2)

with col1:
    fig_cost = px.line(
        x=iterations, y=costs,
        labels={'x': 'Iteration', 'y': 'Cost'},
        title='Cost vs. Iterations'
    )
    st.plotly_chart(fig_cost)

with col2:
    fig_pred = px.line(
        x=iterations, y=predictions,
        labels={'x': 'Iteration', 'y': 'Prediction'},
        title='Prediction Evolution'
    )
    fig_pred.add_hline(y=Y[0][0], line_dash="dash", line_color="red", annotation_text="True Value")
    st.plotly_chart(fig_pred)

# Display final weights and biases
st.subheader('Final Parameters')
st.write('Final Weight Matrix W1:', parameters['W1'])
st.write('Final Bias Vector b1:', parameters['b1'])
st.write('Final Weight Matrix W2:', parameters['W2'])
st.write('Final Bias Vector b2:', parameters['b2'])

# Display training results
st.subheader('Training Results')
st.write('Final Cost:', costs[-1])

# Display final predictions
final_output, _ = forward_propagation(X, parameters)
st.write('Final Predictions:', final_output)

# Prediction comparison visualization
st.subheader('Prediction vs. True Value')
comparison_df = pd.DataFrame({
    'Value': [final_output[0][0], Y[0][0]],
    'Type': ['Prediction', 'True Value']
})
fig_comparison = px.bar(
    comparison_df,
    x='Type',
    y='Value',
    color='Type',
    title='Final Prediction Comparison'
)
st.plotly_chart(fig_comparison)

