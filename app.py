# app.py

import streamlit as st
from neural_network import forward_propagation, backward_propagation

st.title('Manual Neural Network Computation')

# Input data
x = st.number_input('Input (x):', value=0.0)
y = st.number_input('Target Output (y):', value=0.0)

# Parameters
w = st.number_input('Weight (w):', value=0.5)
b = st.number_input('Bias (b):', value=0.0)
learning_rate = st.number_input('Learning Rate:', value=0.1)

# Forward propagation
if st.button('Compute Forward Propagation'):
    output, net_input = forward_propagation(x, w, b)
    st.write(f'Net Input: {net_input}')
    st.write(f'Output: {output}')

    # Backward propagation
    d_loss_d_w, d_loss_d_b, error = backward_propagation(x, y, output, net_input)
    st.write(f'Error: {error}')
    st.write(f'Gradient wrt Weight: {d_loss_d_w}')
    st.write(f'Gradient wrt Bias: {d_loss_d_b}')

    # Update parameters
    if st.button('Update Parameters'):
        w -= learning_rate * d_loss_d_w
        b -= learning_rate * d_loss_d_b
        st.write(f'Updated Weight: {w}')
        st.write(f'Updated Bias: {b}')
