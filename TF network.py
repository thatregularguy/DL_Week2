import numpy as np
import tensorflow as tf
from sklearn.datasets import load_breast_cancer
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Load the Breast Cancer Wisconsin dataset
breast_cancer = load_breast_cancer()
data = breast_cancer.data
target = breast_cancer.target

# Split the data into train and test sets
train_inputs, test_inputs, train_targets, test_targets = train_test_split(data, target, test_size=0.2, random_state=42)

# Scale the data using StandardScaler
scaler = StandardScaler()
train_inputs = scaler.fit_transform(train_inputs)
test_inputs = scaler.transform(test_inputs)

# We don't need the backward propagation here, because
# GradientTape stores all the gradients for the updates

class DenseLayer(tf.Module):
    def __init__(self, output_size, activation=None, dropout_rate=0.0):
        super(DenseLayer, self).__init__()
        self.inputs = None  # Store the inputs
        self.output_size = output_size  # Neurons count
        self.activation = activation
        self.W = None  # Weights
        self.b = None  # Biases
        self.dropout_rate = dropout_rate
        self.dropout_mask = None

    def initialize_weights(self, input_size):
        # Weights and biases using Xavier initialization
        xavier_stddev = np.sqrt(2 / (input_size + self.output_size))
        self.W = tf.Variable(np.random.randn(input_size, self.output_size).astype(np.float64) * xavier_stddev,
                             dtype=tf.float64)
        self.b = tf.Variable(np.random.randn(self.output_size).astype(np.float64) * xavier_stddev, dtype=tf.float64)

    def forward(self, inputs, training=True):
        if self.W is None:  # For the very first time
            self.initialize_weights(inputs.shape[-1])

        self.inputs = inputs  # Store the inputs
        linear_output = tf.matmul(inputs, self.W) + self.b

        # Dropout
        if training and self.dropout_rate > 0:
            self.dropout_mask = tf.cast(tf.random.uniform(linear_output.shape) > self.dropout_rate, tf.float32)
            linear_output *= self.dropout_mask / (1 - self.dropout_rate)

        if self.activation is None:  # Linear Regression
            return linear_output
        elif self.activation == 'sigmoid':  # Sigmoid activation (0,1)
            return tf.sigmoid(linear_output)
        elif self.activation == 'relu':  # ReLU = max(0,x)
            return tf.nn.relu(linear_output)
        else:
            raise ValueError("Supported activations: None for linear, 'sigmoid', 'relu'")

class Network(tf.Module):
    def __init__(self):
        super(Network, self).__init__()
        self.layers = []

    def add_layer(self, layer):  # Adding a layer object to the network
        self.layers.append(layer)

    def forward(self, inputs, training=True):
        for layer in self.layers:  # Feedforward for all the layers
            inputs = layer.forward(inputs, training)
        return inputs

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

        return best_threshold

    def call(self, inputs, targets=None, num_epochs=50000, learning_rate=1e-3, train=True, problem_type='regression'):
        if train:
            for epoch in range(num_epochs):
                # Gradients calculation
                with tf.GradientTape() as tape:
                    outputs = self.forward(inputs, train)  # Predictions
                    if problem_type == 'regression':  # MSE for the regression problem
                        loss = tf.reduce_mean(tf.square(outputs - targets)) / 2
                    elif problem_type == 'classification':  # Cross Entropy Loss for the binary classification problem
                        loss = -tf.reduce_mean(targets * tf.math.log(outputs) + (1 - targets) * tf.math.log(1 - outputs))
                    else:
                        raise ValueError("Invalid problem_type. Supported values: 'regression' or 'classification'")

                if (epoch + 1) % 100 == 0:
                    print(f"Epoch: {epoch+1}, Loss: {loss}")

                # Saving the gradients of weights and biases for the entire network
                # gradients object has a type tuple
                gradients = tape.gradient(loss, self.trainable_variables)

                # Updating the gradients of weights and biases because they are the trainable variables
                # For three layers there will be 3 weights and biases [w, b, w, b, w, b]
                for layer, i in zip(reversed(self.layers), range(len(gradients)-1, 0, -2)):
                    layer.W.assign(layer.W - learning_rate * gradients[i-1])  # Updating the weights
                    layer.b.assign(layer.b - learning_rate * gradients[i])  # Updating the biases
        else:  # Test instance
            outputs = self.forward(inputs, train)
            return outputs


# Create the network and add layers
network = Network()
network.add_layer(DenseLayer(5, activation='relu'))
network.add_layer(DenseLayer(10, activation='relu'))
network.add_layer(DenseLayer(1, activation='sigmoid'))


# Train the network
network.call(train_inputs, train_targets, num_epochs=5000, learning_rate=1, train=True, problem_type='classification')

# Calculate the best threshold
best_threshold = network.find_best_threshold(train_inputs, train_targets)

# Testing the network
predictions = network.call(test_inputs, train=False)

# Post-processing
class_labels = np.where(predictions >= best_threshold, 1, 0)

# Calculate accuracy
accuracy = np.mean(class_labels == test_targets)
print(f"Accuracy: {accuracy}")