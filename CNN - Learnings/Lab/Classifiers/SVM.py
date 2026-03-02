"""
SVM Classifier on Wine Dataset
This script trains and evaluates an SVM classifier on the Wine dataset.
"""
import numpy as np
from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import argparse
from sklearn.preprocessing import StandardScaler

def main(C=1.0, kernel='rbf', test_size=0.3, random_state=42, normalize=False, save_results=None, save_plot=None, verbose=True):
    # Load dataset (Wine dataset)
    wine = load_wine()
    X, y = wine.data, wine.target

    # Normalize features if requested
    if normalize:
        scaler = StandardScaler()
        X = scaler.fit_transform(X)

    # Print dataset head (first 5 samples with all columns and column names)
    print("\nDataset head (first 5 samples):")
    print("Feature names:")
    print(wine.feature_names if hasattr(wine, 'feature_names') else [f'col_{i}' for i in range(X.shape[1])])
    print("Features:")
    print(X[:5, :])
    print("Labels:")
    print(y[:5])

    # Split dataset
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )

    # Define SVM classifier
    svm = SVC(C=C, kernel=kernel, random_state=random_state)

    # Train the model
    svm.fit(X_train, y_train)

    # Test the model
    y_pred = svm.predict(X_test)

    # Evaluate
    acc = accuracy_score(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred)
    report = classification_report(y_test, y_pred)

    print("Accuracy:", acc)
    print("\nConfusion Matrix:\n", cm)
    print("\nClassification Report:\n", report)

    # Save results to file if requested
    if save_results:
        with open(save_results, "w") as f:
            f.write(f"Accuracy: {acc}\n\n")
            f.write(f"Confusion Matrix:\n{cm}\n\n")
            f.write(f"Classification Report:\n{report}\n")

    # Plot confusion matrix
    plt.figure(figsize=(6, 4))
    plt.imshow(cm, interpolation="nearest", cmap="Blues")
    plt.title("Confusion Matrix")
    plt.colorbar()
    tick_marks = np.arange(len(wine.target_names))
    plt.xticks(tick_marks, wine.target_names, rotation=45)
    plt.yticks(tick_marks, wine.target_names)
    plt.ylabel("True label")
    plt.xlabel("Predicted label")
    plt.tight_layout()
    if save_plot:
        plt.savefig(save_plot)
        print(f"Confusion matrix plot saved to {save_plot}")
    plt.show()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="SVM Classifier on Wine Dataset")
    parser.add_argument('--C', type=float, default=1.0, help='Regularization parameter for SVM')
    parser.add_argument('--kernel', type=str, default='rbf', help='Kernel type for SVM')
    parser.add_argument('--test_size', type=float, default=0.3, help='Test set size (fraction)')
    parser.add_argument('--random_state', type=int, default=42, help='Random state for train/test split')
    parser.add_argument('--normalize', action='store_true', help='Normalize features')
    parser.add_argument('--save_results', type=str, help='File to save results')
    parser.add_argument('--save_plot', type=str, help='File to save confusion matrix plot')
    parser.add_argument('--verbose', action='store_true', help='Print detailed output')
    args = parser.parse_args()
    main(C=args.C, kernel=args.kernel, test_size=args.test_size, random_state=args.random_state,
         normalize=args.normalize, save_results=args.save_results, save_plot=args.save_plot, verbose=args.verbose)
