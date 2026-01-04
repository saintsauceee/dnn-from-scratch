import numpy as np

class Perceptron:
    def __init__(self, n_features, lr, n_epochs):
        self.lr = lr
        self.n_epochs = n_epochs

        # small random weights (normal distribution)
        self.w = np.random.normal(loc=0.0, scale=0.01, size=(n_features,))
        self.b = 0.0
    
    def fit(self, X, y):
        """
        X: shape (n_samples, n_features)
        y: shape (n_samples,) with labels 0 or 1
        """
        # convert {0, 1} -> {-1, 1}
        y_signed = np.where(y == 0, -1, 1)

        for _ in range(self.n_epochs):
            for xi, target in zip(X, y_signed):
                activation = np.dot(xi, self.w) + self.b

                # if missclassified (different signs or 0)
                if target * activation <= 0:
                    self.w += self.lr * target * xi
                    self.b += self.lr * target

    def predict(self, X):
        """
        Returns labels 0 or 1
        """
        scores = X @ self.w + self.b
        return (scores > 0).astype(int)

if __name__ == "main":
    # XOR
    X = [[0, 0], [0, 1], [1, 0], [1, 1]]
    y = [0, 1, 1, 0]

    Perceptron(2, 0.01, 2)