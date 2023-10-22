import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sys
from prediction import y_hat_
from training import fit_

if len(sys.argv[1:]) == 0:
    data = pd.read_csv('data.csv')
    x = np.array(data['km']).reshape(-1, 1)
    y = np.array(data['price']).reshape(-1, 1)
    print (x, y)
    theta = fit_(x, y)
    y_hat = y_hat_(x, theta)
    plt.scatter(x, y)
    plt.plot(x, y_hat, "r-")
    plt.show()
else:
    print("Usage: ./python3 MyLinearRegression.py")
