"""
test program
"""

import numpy as np
import pandas as pd
from MyLinearRegression import MyLinearRegression as MyLR
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error

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
        y_hat = my_lr.predict_(x)
        plt.plot(x, y_hat)
        print(theta[0] + num2 * theta[1])
        plt.scatter(x, y)
        plt.savefig('image.png')
        print(my_lr.mse_(y, y_hat))
        print(mean_squared_error(y, y_hat))
    else:
        print("Usage: ./python3 test.py")

if __name__ == "__main__":
    import sys
    main()
