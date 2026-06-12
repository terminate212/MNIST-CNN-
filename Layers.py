from abc import ABC, abstractmethod
import numpy as np

class BaseLayer(ABC):
    @abstractmethod
    def forward(self, x : np.ndarray) -> np.array:
        pass

    @abstractmethod
    def backward(self, upstream_delta : np.ndarray, batch_size : int) -> np.ndarray:
        pass

class DenseLayer(BaseLayer):
    def __init__(self, input_size, output_size):
        self.input_size = input_size
        self.output_size = output_size
        self.weights = np.random.randn(output_size, input_size) * np.sqrt(2. / input_size)
        self.biases = np.zeros((output_size, 1))
        self.z_neurons = None
        self.input_cache = None

    def forward(self, inputs):
        self.input_cache = np.copy(inputs)
        self.z_neurons = np.dot(self.weights, inputs) + self.biases
        return self.z_neurons

    def backward(self, upstream_delta, batch_size):
        self.weight_grad = np.dot(upstream_delta, self.input_cache.T) / batch_size
        self.bias_grad = np.sum(upstream_delta, axis=1, keepdims=True) / batch_size
        return np.dot(self.weights.T, upstream_delta)

    def update(self, learning_rate):
        self.weights -= learning_rate * self.weight_grad
        self.biases -= learning_rate * self.bias_grad

# class ConvLayer(BaseLayer):
#     pass

# class MaxPoolLayer(BaseLayer):
#     pass

# class FlattenLayer(BaseLayer):
#     pass

class ActivationLayer(BaseLayer):
    def __init__(self, activation_function):
        self.activation_function = {
            "ReLU" : self.__ReLU,
            "softmax" : self.__softmax,
        }[activation_function]

        self.backward = {
            "ReLU" : self.__ReLU_grad,
            "softmax" : self.__softmax_grad,
        }[activation_function]

        self.z_neurons = None

    def forward(self, inputs):
        return self.activation_function(inputs)

    def __ReLU(self, z):
        self.z_neurons = np.copy(z)
        return np.maximum(0, z)
    
    def __ReLU_grad(self):
        return (self.z_neurons > 0).astype(float)

    def __softmax(self, logits):
        max_logit = np.max(logits, axis = 0, keepdims = True)
        exp = np.exp(np.subtract(logits, max_logit))
        return exp / np.sum(exp, axis = 0, keepdims = True)
    
        
    def __softmax_grad():
        # pass through from loss class
        
        pass

if __name__ == "__main__":
    layer = DenseLayer(10, 10)
    print("sdhskdh")
    print(layer.z_neurons)