class SequentialModel:
    def __init__(self):
        self.layers = []

    def add(self, layer):
        self.layers.append(layer)

    def forward(self, inputs):
        output = inputs
        for layer in self.layers:
            output = layer.forward(output)
        return output

    def backward(self, d_values, learning_rate):
        gradient = d_values
        for layer in reversed(self.layers):
            gradient = layer.backward(gradient, learning_rate)
