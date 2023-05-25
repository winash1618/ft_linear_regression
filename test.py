"""
test program
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error
from MyLinearRegression import MyLinearRegression as MyLR

def main() -> None:
    """
        This function is made to shorten the length of __name__ == "__main__" condition
    """
    if len(sys.argv[1:]) == 0:
        num2 = float(input("Enter the mileage of the Car you want to buy: "))
        data = pd.read_csv('data.csv')
        x = np.array(data['km']).reshape(-1, 1)
        y = np.array(data['price']).reshape(-1, 1)
        my_lr = MyLR()
        theta = my_lr.fit_(x, y)
        print("theta0 = ", theta[0][0])
        print("theta1 = ", theta[1][0])
        y_hat = my_lr.predict_(x)
        plt.scatter(x, y)
        plt.plot(x, y_hat, "r-")
        plt.savefig('image.png')
        print("Estimated price: ", theta[0][0] + theta[1][0] * num2)
        print("mean squared error for My Linear Regression model: ", my_lr.mse_(y, y_hat))
        print("Mean squared error for My Linear Regression model: ", mean_squared_error(y, y_hat))
    else:
        print("Usage: ./python3 test.py")

if __name__ == "__main__":
    import sys
    main()
