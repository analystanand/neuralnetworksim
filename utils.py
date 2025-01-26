import numpy as np
import matplotlib.pyplot as plt

def plot_loss(loss_history):
    plt.plot(loss_history)
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training Loss Over Time')
    plt.show()

def normalize_data(data):
    return (data - np.mean(data)) / np.std(data)
