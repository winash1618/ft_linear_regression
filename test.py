"""
test program
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error
from MyLinearRegression import MyLinearRegression as MyLR

def normalize(X):
    X_normalized = (X - np.min(X, axis=0)) / (np.max(X, axis=0) - np.min(X, axis=0))
    return X_normalized

def main() -> None:
    """
        This function is made to shorten the length of __name__ == "__main__" condition
    """
    if len(sys.argv[1:]) == 0:
        num2 = float(input("Enter the mileage of the Car you want to buy: "))
        data = pd.read_csv('data.csv')
        x = np.array(data['km']).reshape(-1, 1)
        nx = normalize(x)
        y = np.array(data['price']).reshape(-1, 1)
        ny = normalize(y)
        print(x, y)
        my_lr = MyLR()
        theta = my_lr.fit_(nx, ny)
        x_min = np.min(x)
        x_max = np.max(x)
        x_range = x_max - x_min
        y_mean = np.mean(y)
        y_std = np.std(y)

        theta_0_original = y_mean - theta[0] * y_std / x_range
        theta_1_original = theta[1] * y_std / x_range
        my_lr.thetas[0] = theta_0_original
        my_lr.thetas[1] = theta_1_original
        print(theta)
        y_hat = my_lr.predict_(x)
        print(y_hat)
        plt.plot(x, y_hat)
        # result = theta[0] + (num2 - np.min(x)) / (np.max(x) - np.min(x)) * theta[1]
        # print( (result * (np.max(y) - np.min(y))) + np.min(y))
        plt.scatter(x, y)
        plt.savefig('image.png')
        print(my_lr.mse_(ny, y_hat))
        print(mean_squared_error(ny, y_hat))
    else:
        print("Usage: ./python3 test.py")

if __name__ == "__main__":
    import sys
    main()
