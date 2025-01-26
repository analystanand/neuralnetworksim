# Neural Network Simulator

A simple neural network simulation tool implemented in Python that allows users to create, train, and experiment with various neural network architectures.

## Features

- Custom neural network architecture creation
- Support for various activation functions
- Training with backpropagation
- Real-time visualization of network performance
- Data preprocessing utilities

## Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/neuralnetworksim.git

# Navigate to the project directory
cd neuralnetworksim

# Install dependencies
pip install -r requirements.txt
```

## Usage

```python
from neuralnetworksim import NeuralNetwork

# Create a neural network with 3 layers
nn = NeuralNetwork([2, 3, 1])

# Train the neural network with training data
training_data = [
    ([0, 0], [0]),
    ([0, 1], [1]),
    ([1, 0], [1]),
    ([1, 1], [0])
]
nn.train(training_data, epochs=1000, learning_rate=0.1)

# Make a prediction
output = nn.predict([1, 0])
print(f"Prediction for input [1, 0]: {output}")
```

## Project Structure

```
neuralnetworksim/
├── src/           # Source code files
├── tests/         # Test files
├── examples/      # Example implementations
└── docs/          # Documentation
```

## Contributing

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add some amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Contact

Parma - parma@example.com
Project Link: [https://github.com/yourusername/neuralnetworksim](https://github.com/yourusername/neuralnetworksim)
