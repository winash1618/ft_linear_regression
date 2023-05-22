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
        # num2 = float(input("Enter the mileage of the Car you want to buy: "))
        data = pd.read_csv('data.csv')
        x = np.array(data['km']).reshape(-1, 1)
        y = np.array(data['price']).reshape(-1, 1)
        my_lr = MyLR()
        theta = my_lr.fit_(x, y)
        print(theta)
        y_hat = my_lr.predict_(x)
        print(y_hat)
        plt.scatter(x, y)
        plt.plot(x, y_hat)
        plt.savefig('image.png')
        print(my_lr.mse_(y, y_hat))
        print(mean_squared_error(y, y_hat))
    else:
        print("Usage: ./python3 test.py")

if __name__ == "__main__":
    import sys
    main()
