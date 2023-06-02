import numpy as np
from sklearn.preprocessing import StandardScaler

class Layer:
    def __init__(self, output_size, activation=None, dropout_rate=0.0):
        self.output_size = output_size  # Neurons in layer
        self.W = None  # Weights
        self.b = None  # Biases
        self.inputs = None  # Required for saving the inputs of layers
        self.activation = activation  # Activation functions: None for linear, sigmoid or relu
        # Dropout
        self.dropout_rate = dropout_rate
        self.dropout_mask = None  # Required for saving the dropped neurons
        # Adam optimizer
        self.v_w = None  # First moment vector for weights
        self.r_w = None  # Second moment vector for weights
        self.v_b = None  # First moment vector for biases
        self.r_b = None  # Second moment vector for biases
        self.t = 0  # Time step counter

    def initialize_weights(self, input_size):
        # Normal Xavier initialization
        xavier_stddev = np.sqrt(2 / (input_size + self.output_size))
        # Weights and biases initialization
        self.W = np.random.randn(input_size, self.output_size) * xavier_stddev
        self.b = np.random.randn(self.output_size) * xavier_stddev
        # Moments' initialization
        self.v_w = np.zeros_like(self.W)
        self.r_w = np.zeros_like(self.W)
        self.v_b = np.zeros_like(self.b)
        self.r_b = np.zeros_like(self.b)

    def forward(self, inputs, train=True):
        self.inputs = inputs
        input_size = inputs.shape[-1]

        if self.W is None:  # Required for the first initialization
            self.initialize_weights(input_size)

        linear_output = np.dot(inputs, self.W) + self.b  # Layer's linear combination

        if train and self.dropout_rate > 0:  # Dropout with binomial selection of neurons
            self.dropout_mask = np.random.binomial(1, 1 - self.dropout_rate, size=linear_output.shape)
            linear_output *= self.dropout_mask / (1 - self.dropout_rate)

        if self.activation is None:  # Linear Regression
            return linear_output
        elif self.activation == 'sigmoid':  # Sigmoid Activation
            return 1 / (1 + np.exp(-linear_output))
        elif self.activation == 'relu':  # ReLU Activation
            return np.maximum(0, linear_output)
        else:
            raise ValueError("Supported activations: None for linear, 'sigmoid', 'relu'")

    def backward(self, grad_output, learning_rate, beta1=0.9, beta2=0.999, epsilon=1e-10):
        # Activation's derivative to its input
        # Important for using in chain rule
        if self.activation is None:
            activation_derivative = 1
        elif self.activation == 'sigmoid':
            sigmoid_output = 1 / (1 + np.exp(-self.inputs))
            activation_derivative = sigmoid_output * (1 - sigmoid_output)
        elif self.activation == 'relu':
            activation_derivative = np.where(self.inputs > 0, 1, 0)
        else:
            raise ValueError("Supported activations: None for linear, 'sigmoid', 'relu'")

        # For the case of dropout
        if self.dropout_mask is not None:
            grad_output *= self.dropout_mask / (1 - self.dropout_rate)

        # Adam optimization
        self.t += 1

        # Gradients calculation
        grad_inputs = np.dot(grad_output, self.W.T) * activation_derivative
        grad_W = np.dot(self.inputs.T, grad_output)
        grad_b = np.sum(grad_output, axis=0)

        # Weights update using Adam optimizer
        self.v_w = beta1 * self.v_w + (1 - beta1) * grad_W
        self.r_w = beta2 * self.r_w + (1 - beta2) * (grad_W ** 2)
        v_w_hat = self.v_w / (1 - beta1 ** self.t)
        r_w_hat = self.r_w / (1 - beta2 ** self.t)
        self.W -= learning_rate * v_w_hat / (np.sqrt(r_w_hat) + epsilon)

        # Biases update using Adam optimizer
        self.v_b = beta1 * self.v_b + (1 - beta1) * grad_b
        self.r_b = beta2 * self.r_b + (1 - beta2) * (grad_b ** 2)
        v_b_hat = self.v_b / (1 - beta1 ** self.t)
        r_b_hat = self.r_b / (1 - beta2 ** self.t)
        self.b -= learning_rate * v_b_hat / (np.sqrt(r_b_hat) + epsilon)

        return grad_inputs


class Network:
    def __init__(self):
        self.layers = []

    def add_layer(self, layer):  # Adding a layer object to the network
        self.layers.append(layer)

    def forward(self, inputs, train=True):
        for layer in self.layers:  # Feedforward for all the layers
            inputs = layer.forward(inputs, train)
        return inputs

    def backward(self, grad_output, learning_rate, beta1=0.9, beta2=0.999, epsilon=1e-10):
        for layer in reversed(self.layers):  # Backward for all the layers with Adam
            grad_output = layer.backward(grad_output, learning_rate, beta1, beta2, epsilon)
        return grad_output

    def find_best_threshold(self, inputs, targets):
        predictions = self.call(inputs, train=False)
        best_threshold = 0.0
        best_f1_score = 0.0

        # Loop through different thresholds to find the best one
        for threshold in np.arange(0.1, 1.0, 0.1):
            class_labels = np.where(predictions >= threshold, 1, 0)
            f1 = f1_score(targets, class_labels)
            if f1 > best_f1_score:
                best_f1_score = f1
                best_threshold = threshold

        return best_threshold, best_f1_score

    def call(self, inputs, targets=None, num_epochs=50000, learning_rate=1e-3, train=True, problem_type='regression',
             beta1=0.9, beta2=0.999, epsilon=1e-8):

        if train:  # Training for a train set
            for epoch in range(num_epochs):
                outputs = self.forward(inputs, train)
                if problem_type == 'regression':  # MSE for a regression problem
                    loss = np.mean((outputs - targets) ** 2) / 2
                    grad_output = (outputs - targets) / len(targets)
                elif problem_type == 'classification':  # CEL for a classification problem
                    loss = -np.mean(targets * np.log(outputs) + (1 - targets) * np.log(1 - outputs))
                    grad_output = (-targets / outputs + (1 - targets) / (1 - outputs)) / len(targets)
                else:
                    raise ValueError("Invalid problem_type. Supported values: 'regression' or 'classification'")

                self.backward(grad_output, learning_rate, beta1, beta2, epsilon)

                if epoch % 100 == 0:
                    print(f"Epoch {epoch}: Loss = {loss:.8f}")
        else:  # Prediction for a test set
            outputs = self.forward(inputs, train)
            return outputs

np.random.seed(42)
X = np.random.randn(300, 5)  # Generate 5-dimensional input data
y = np.random.randint(0, 2, size=(300,1))  # Generate binary labels (0 or 1)

scaler = StandardScaler()
X = scaler.fit_transform(X)

# Split data into train and test sets for classification
split_idx = int(0.8 * len(X))
X_train, y_train = X[:split_idx], y[:split_idx]
X_test, y_test = X[split_idx:], y[split_idx:]

# Create and configure the network for classification
network_classification = Network()
network_classification.add_layer(Layer(output_size=5, activation='relu'))
network_classification.add_layer(Layer(output_size=10, activation='relu'))
network_classification.add_layer(Layer(output_size=1, activation='sigmoid'))

# Train the network for classification
network_classification.call(X_train, y_train, num_epochs=100000, learning_rate=2e-5, train=True, problem_type='classification')

# Evaluate the network on the test set for classification
test_predictions = network_classification.call(X_test, train=False)

test_predictions = np.round(test_predictions >= 0.5).astype(int)
accuracy = np.mean(test_predictions == y_test)
print("Classification Accuracy:", accuracy)