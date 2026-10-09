import numpy as np

class DenseLayer:
    def __init__(self, input_dim, output_dim):
        self.weights = np.random.randn(input_dim, output_dim) * np.sqrt(2.0 / input_dim)
        self.biases = np.zeros((1, output_dim))
        self.dW, self.db = None, None

    def forward(self, inputs):
        self.inputs = inputs
        return np.dot(inputs, self.weights) + self.biases

    def backward(self, d_values, learning_rate):
        self.dW = np.dot(self.inputs.T, d_values)
        self.db = np.sum(d_values, axis=0, keepdims=True)
        d_inputs = np.dot(d_values, self.weights.T)
        
        self.weights -= learning_rate * self.dW
        self.biases -= learning_rate * self.db
        return d_inputs

class ReLUActivation:
    def forward(self, inputs):
        self.inputs = inputs
        return np.maximum(0, inputs)

    def backward(self, d_values, learning_rate=None):
        self.d_inputs = d_values.copy()
        self.d_inputs[self.inputs <= 0] = 0
        return self.d_inputs
