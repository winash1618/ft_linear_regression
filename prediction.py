import numpy as np

def predict_(x, thetas):
    """_summary_

    Args:
        x (numpy.ndarray): a vector of dimension m * 1
    """
    if not isinstance(x, np.ndarray) or not isinstance(thetas, np.ndarray):
        return None
    if x.size == 0 or thetas.size == 0:
        return None
    new_x = np.insert(x, 0, np.array([1]), axis=1)
    return new_x @ thetas

def main() -> None:
    """
        This function is made to shorten the length of __name__ == "__main__" condition
    """
    if len(sys.argv[1:]) == 0:
        num2 = float(input("Enter the mileage of the Car you want to buy: "))
        data = pd.read_csv('data.csv')
        x = np.array(data['km']).reshape(-1, 1)
        y = np.array(data['price']).reshape(-1, 1)
        theta = np.array([[0], [0]])
        print (x, y)
        print("theta0 = ", theta[0][0])
        print("theta1 = ", theta[1][0])
        print("Estimated price: ", theta[0][0] + theta[1][0] * num2)
    else:
        print("Usage: ./python3 MyLinearRegression.py")


if __name__ == "__main__":
    import pandas as pd
    import sys
    main()
