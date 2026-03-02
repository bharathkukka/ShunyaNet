import numpy as np
from sklearn.datasets import make_moons
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from twoLayerNN import TwoLayerNN

# Load toy dataset
X, y = make_moons(n_samples=1000, noise=0.2, random_state=42)
X = X.T  # shape: (2, n_samples)
y = y.reshape(-1, 1)

# One-hot encode labels
enc = OneHotEncoder(sparse_output=False)
Y = enc.fit_transform(y).T  # shape: (2, n_samples)

# Split into train/test
X_train, X_test, Y_train, Y_test = train_test_split(X.T, Y.T, test_size=0.2, random_state=42)
X_train, X_test = X_train.T, X_test.T
Y_train, Y_test = Y_train.T, Y_test.T

# Initialize and train the network
nn = TwoLayerNN(n_x=2, n_h=10, n_y=2, lr=0.1)
nn.train(X_train, Y_train, epochs=1000)

# Predict and evaluate
metrics = nn.evaluate(X_test, Y_test)
print("Test Metrics:")
for k, v in metrics.items():
    print(f"{k.capitalize()}: {v:.4f}")
