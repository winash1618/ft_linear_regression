"""
My Linear Regression
"""

import numpy as np

class MyLinearRegression():
    """
    Description:
    My personnal linear regression class to fit like a boss.
    """

    def __init__(self, thetas=np.array([[0], [0]]), alpha1=0.01, alpha2=0.0000000001, max_iter=10000):
        self.alpha1 = alpha1
        self.alpha2 = alpha2
        self.max_iter = max_iter
        self.thetas = thetas

    def fit_(self, x, y):
        """Fits the model to the training dataset contained in x and y.

        Args:
            x (numpy.ndarray): a vector of dimension m * 1
            y (numpy.ndarray): a vector of dimension m * 1
        """
        if not isinstance(x, np.ndarray) or not isinstance(y, np.ndarray):
            return None
        if x.size == 0 or y.size == 0 or self.thetas.size == 0:
            return None
        if x.shape != y.shape:
            return None
        m, _ = x.shape
        i = 0
        theta0 = self.thetas[0][0]
        theta1 = self.thetas[1][0]
        
        while i < self.max_iter:
            X = np.insert(x, 0, np.array([1]), axis=1)
            grad = (1 / m) * np.dot(X.transpose(), (np.dot(X, self.thetas) - y))
            theta0 = theta0 - self.alpha1 * grad[0][0]
            theta1 = theta1 - self.alpha2 * grad[1][0]
            self.thetas = np.array([[theta0], [theta1]])
            i += 1
            
        return self.thetas

    def mse_(self, y, y_hat):
        """_summary_

        Args:
            y (numpy.ndarray): a vector of dimension m * 1
            y_hat (numpy.ndarray): a vector of dimension m * 1
        """
        if not isinstance(y, np.ndarray) and not isinstance(y_hat, np.ndarray):
            return None
        if y.shape != y_hat.shape:
            return None
        if y.size == 0 or y_hat.size == 0:
            return None
        return np.squeeze((1 / len(y)) * ((y_hat - y).transpose() @ (y_hat - y)))

    def predict_(self, x):
        """_summary_

        Args:
            x (numpy.ndarray): a vector of dimension m * 1
        """
        if not isinstance(x, np.ndarray) or not isinstance(self.thetas, np.ndarray):
            return None
        if x.size == 0 or self.thetas.size == 0:
            return None
        new_x = np.insert(x, 0, np.array([1]), axis=1)
        return new_x @ self.thetas

    def loss_elem_(self, y, y_hat):
        """_summary_

        Args:
            y (numpy.ndarray): a vector of dimension m * 1
            y_hat (numpy.ndarray): a vector of dimension m * 1
        """
        return np.square(y_hat - y)

    def loss_(self, y, y_hat):
        """_summary_

        Args:
            y (numpy.ndarray): a vector of dimension m * 1
            y_hat (numpy.ndarray): a vector of dimension m * 1
        """
        if not isinstance(y, np.ndarray) and not isinstance(y_hat, np.ndarray):
            return None
        if y.shape != y_hat.shape:
            return None
        loss = self.loss_elem_(y, y_hat)
        return np.sum(loss) / (2 * len(loss))

def main() -> None:
    """
        This function is made to shorten the length of __name__ == "__main__" condition
    """
    if len(sys.argv[1:]) == 0:
        data = pd.read_csv('data.csv')
        x = np.array(data['km']).reshape(-1, 1)
        y = np.array(data['price']).reshape(-1, 1)
        print (x, y)
        my_lr = MyLinearRegression()
        theta = my_lr.fit_(x, y)
        print(theta)
    else:
        print("Usage: ./python3 MyLinearRegression.py")


if __name__ == "__main__":
    import pandas as pd
    import sys
    main()
