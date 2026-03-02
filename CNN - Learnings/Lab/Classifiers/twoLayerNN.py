import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_moons
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# Activation Functions
def relu(Z): return np.maximum(0, Z)
def relu_derivative(Z): return (Z > 0).astype(float)
def softmax(Z):
    expZ = np.exp(Z - np.max(Z, axis=0, keepdims=True))
    return expZ / np.sum(expZ, axis=0, keepdims=True)

def compute_loss(Y_hat, Y):
    m = Y.shape[1]
    return -np.sum(Y * np.log(Y_hat + 1e-9)) / m

# Two-Layer Neural Network
class TwoLayerNN:
    def __init__(self, n_x, n_h, n_y, lr=0.01, seed=None):
        if seed is not None:
            np.random.seed(seed)
        self.W1 = np.random.randn(n_h, n_x) * np.sqrt(2. / n_x)
        self.b1 = np.zeros((n_h, 1))
        self.W2 = np.random.randn(n_y, n_h) * np.sqrt(2. / n_h)
        self.b2 = np.zeros((n_y, 1))
        self.lr = lr

    def forward(self, X):
        self.Z1 = np.dot(self.W1, X) + self.b1
        self.A1 = relu(self.Z1)
        self.Z2 = np.dot(self.W2, self.A1) + self.b2
        self.A2 = softmax(self.Z2)
        return self.A2

    def backward(self, X, Y):
        m = X.shape[1]
        dZ2 = self.A2 - Y
        dW2 = (1/m) * np.dot(dZ2, self.A1.T)
        db2 = (1/m) * np.sum(dZ2, axis=1, keepdims=True)
        dA1 = np.dot(self.W2.T, dZ2)
        dZ1 = dA1 * relu_derivative(self.Z1)
        dW1 = (1/m) * np.dot(dZ1, X.T)
        db1 = (1/m) * np.sum(dZ1, axis=1, keepdims=True)
        # Update
        self.W1 -= self.lr * dW1
        self.b1 -= self.lr * db1
        self.W2 -= self.lr * dW2
        self.b2 -= self.lr * db2

    def train(self, X, Y, epochs=1000, verbose=True):
        losses = []
        for i in range(epochs):
            Y_hat = self.forward(X)
            loss = compute_loss(Y_hat, Y)
            losses.append(loss)
            self.backward(X, Y)
            if verbose and (i % 100 == 0 or i == epochs-1):
                print(f"Epoch {i}, Loss: {loss:.4f}")
        return losses

    def predict(self, X):
        Y_hat = self.forward(X)
        return np.argmax(Y_hat, axis=0)

    def evaluate(self, X, Y_true):
        preds = self.predict(X)
        true_labels = np.argmax(Y_true, axis=0)
        acc = accuracy_score(true_labels, preds)
        prec = precision_score(true_labels, preds, average='weighted', zero_division=0)
        rec = recall_score(true_labels, preds, average='weighted', zero_division=0)
        f1 = f1_score(true_labels, preds, average='weighted', zero_division=0)
        return {'accuracy': acc, 'precision': prec, 'recall': rec, 'f1': f1}
