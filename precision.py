import numpy as np
import pandas as pd
import sys
from sklearn.metrics import mean_squared_error
from prediction import y_hat_
from training import fit_

def loss_elem_(y, y_hat):
    """_summary_

    Args:
        y (numpy.ndarray): a vector of dimension m * 1
        y_hat (numpy.ndarray): a vector of dimension m * 1
    """
    return np.square(y_hat - y)

def loss_(y, y_hat):
    """_summary_

    Args:
        y (numpy.ndarray): a vector of dimension m * 1
        y_hat (numpy.ndarray): a vector of dimension m * 1
    """
    if not isinstance(y, np.ndarray) and not isinstance(y_hat, np.ndarray):
        return None
    if y.shape != y_hat.shape:
        return None
    loss = loss_elem_(y, y_hat)
    return np.sum(loss) / len(loss)

def main() -> None:
    """
        This function is made to shorten the length of __name__ == "__main__" condition
    """
    if len(sys.argv[1:]) == 0:
        data = pd.read_csv('data.csv')
        x = np.array(data['km']).reshape(-1, 1)
        y = np.array(data['price']).reshape(-1, 1)
        print (x, y)
        theta = fit_(x, y)
        y_hat = y_hat_(x, theta)
        print("MSE: ", mean_squared_error(y, y_hat))
        print("My MSE: ", loss_(y, y_hat))
       
    else:
        print("Usage: ./python3 MyLinearRegression.py")

if __name__ == "__main__":
    main()