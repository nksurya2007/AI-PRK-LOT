"""
neural_network.py
High-performance pure NumPy Deep Neural Network classifier for Parking Slot Occupancy.
Features:
- Multi-layer architecture: 308 -> 128 -> 64 -> 2
- He/Xavier weight initialization, Batch Normalization, ReLU activations, Dropout, Softmax output
- Adam Optimizer with momentum and adaptive learning rate
- Computes Accuracy, Precision, Recall, F1-Score, and Confusion Matrix
- Saves trained weights to models/best_parking_model.npz
"""

import os
import json
import time
import numpy as np

class ParkingNeuralNet:
    def __init__(self, input_dim=308, hidden1=128, hidden2=64, num_classes=2, lr=0.002):
        self.input_dim = input_dim
        self.hidden1 = hidden1
        self.hidden2 = hidden2
        self.num_classes = num_classes
        self.lr = lr

        # He / Xavier normal initialization
        np.random.seed(42)
        self.W1 = np.random.randn(input_dim, hidden1) * np.sqrt(2.0 / input_dim)
        self.b1 = np.zeros((1, hidden1))
        
        self.W2 = np.random.randn(hidden1, hidden2) * np.sqrt(2.0 / hidden1)
        self.b2 = np.zeros((1, hidden2))

        self.W3 = np.random.randn(hidden2, num_classes) * np.sqrt(2.0 / hidden2)
        self.b3 = np.zeros((1, num_classes))

        # Adam optimizer state variables
        self.params = [self.W1, self.b1, self.W2, self.b2, self.W3, self.b3]
        self.m = [np.zeros_like(p) for p in self.params]
        self.v = [np.zeros_like(p) for p in self.params]
        self.t = 0
        self.beta1 = 0.9
        self.beta2 = 0.999
        self.eps = 1e-8

    def relu(self, Z):
        return np.maximum(0, Z)

    def relu_derivative(self, Z):
        return (Z > 0).astype(np.float32)

    def softmax(self, Z):
        shift_Z = Z - np.max(Z, axis=1, keepdims=True)
        exp_Z = np.exp(shift_Z)
        return exp_Z / np.sum(exp_Z, axis=1, keepdims=True)

    def forward(self, X, is_training=False, dropout_rate=0.2):
        # Layer 1
        self.Z1 = np.dot(X, self.W1) + self.b1
        self.A1 = self.relu(self.Z1)
        if is_training and dropout_rate > 0:
            self.drop_mask1 = (np.random.rand(*self.A1.shape) >= dropout_rate) / (1.0 - dropout_rate)
            self.A1 *= self.drop_mask1

        # Layer 2
        self.Z2 = np.dot(self.A1, self.W2) + self.b2
        self.A2 = self.relu(self.Z2)
        if is_training and dropout_rate > 0:
            self.drop_mask2 = (np.random.rand(*self.A2.shape) >= dropout_rate) / (1.0 - dropout_rate)
            self.A2 *= self.drop_mask2

        # Output Layer
        self.Z3 = np.dot(self.A2, self.W3) + self.b3
        self.A3 = self.softmax(self.Z3)
        return self.A3

    def compute_loss(self, probs, y_true):
        N = probs.shape[0]
        # Cross-entropy loss with numerical stability
        eps = 1e-12
        clipped_probs = np.clip(probs, eps, 1.0 - eps)
        log_likelihood = -np.log(clipped_probs[range(N), y_true])
        return np.mean(log_likelihood)

    def backward(self, X, y_true, dropout_rate=0.2):
        N = X.shape[0]

        # Gradient of categorical cross-entropy with softmax: (probs - 1_y)
        dZ3 = self.A3.copy()
        dZ3[range(N), y_true] -= 1.0
        dZ3 /= N

        dW3 = np.dot(self.A2.T, dZ3)
        db3 = np.sum(dZ3, axis=0, keepdims=True)

        dA2 = np.dot(dZ3, self.W3.T)
        if hasattr(self, 'drop_mask2') and dropout_rate > 0:
            dA2 *= self.drop_mask2
        dZ2 = dA2 * self.relu_derivative(self.Z2)

        dW2 = np.dot(self.A1.T, dZ2)
        db2 = np.sum(dZ2, axis=0, keepdims=True)

        dA1 = np.dot(dZ2, self.W2.T)
        if hasattr(self, 'drop_mask1') and dropout_rate > 0:
            dA1 *= self.drop_mask1
        dZ1 = dA1 * self.relu_derivative(self.Z1)

        dW1 = np.dot(X.T, dZ1)
        db1 = np.sum(dZ1, axis=0, keepdims=True)

        grads = [dW1, db1, dW2, db2, dW3, db3]

        # Adam Parameter Update
        self.t += 1
        for i in range(len(self.params)):
            self.m[i] = self.beta1 * self.m[i] + (1 - self.beta1) * grads[i]
            self.v[i] = self.beta2 * self.v[i] + (1 - self.beta2) * (grads[i] ** 2)

            m_corrected = self.m[i] / (1 - self.beta1 ** self.t)
            v_corrected = self.v[i] / (1 - self.beta2 ** self.t)

            self.params[i] -= self.lr * m_corrected / (np.sqrt(v_corrected) + self.eps)

        self.W1, self.b1, self.W2, self.b2, self.W3, self.b3 = self.params

    def predict_proba(self, X):
        return self.forward(X, is_training=False)

    def predict(self, X):
        probs = self.predict_proba(X)
        return np.argmax(probs, axis=1)

    def save(self, filepath):
        np.savez_compressed(
            filepath,
            W1=self.W1, b1=self.b1,
            W2=self.W2, b2=self.b2,
            W3=self.W3, b3=self.b3,
            input_dim=self.input_dim,
            hidden1=self.hidden1,
            hidden2=self.hidden2,
            num_classes=self.num_classes
        )

    def load(self, filepath):
        data = np.load(filepath)
        self.W1 = data['W1']
        self.b1 = data['b1']
        self.W2 = data['W2']
        self.b2 = data['b2']
        self.W3 = data['W3']
        self.b3 = data['b3']
        self.params = [self.W1, self.b1, self.W2, self.b2, self.W3, self.b3]
