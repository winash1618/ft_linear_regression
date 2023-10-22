import numpy as np
import pandas as pd
import sys
from prediction import predict_

def fit_(x, y):
    """Fits the model to the training dataset contained in x and y.

    Args:
        x (numpy.ndarray): a vector of dimension m * 1
        y (numpy.ndarray): a vector of dimension m * 1
    """
    alpha1=0.01
    alpha2=0.0000000001
    if not isinstance(x, np.ndarray) or not isinstance(y, np.ndarray):
        return None
    if x.size == 0 or y.size == 0:
        return None
    if x.shape != y.shape:
        return None
    m, _ = x.shape
    thetas = np.array([[0], [0]])
    i = 0
    theta0 = thetas[0][0]
    theta1 = thetas[1][0]
    
    while i < 10000:
        X = np.insert(x, 0, np.array([1]), axis=1)
        grad = (1 / m) * np.dot(X.transpose(), (np.dot(X, thetas) - y))
        theta0 = theta0 - alpha1 * grad[0][0]
        theta1 = theta1 - alpha2 * grad[1][0]
        thetas = np.array([[theta0], [theta1]])
        i += 1
    return thetas

def main() -> None:
    """
        This function is made to shorten the length of __name__ == "__main__" condition
    """
    if len(sys.argv[1:]) == 0:
        num2 = float(input("Enter the mileage of the Car you want to buy: "))
        data = pd.read_csv('data.csv')
        x = np.array(data['km']).reshape(-1, 1)
        y = np.array(data['price']).reshape(-1, 1)
        print (x, y)
        theta = fit_(x, y)
        print("theta0 = ", theta[0][0])
        print("theta1 = ", theta[1][0])
        print("Estimated price: ", theta[0][0] + theta[1][0] * num2)
    else:
        print("Usage: ./python3 MyLinearRegression.py")


if __name__ == "__main__":
    main()
