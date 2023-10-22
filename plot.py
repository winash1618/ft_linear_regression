import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from prediction import predict_
from training import fit_
import sys

if len(sys.argv[1:]) == 0:
    data = pd.read_csv('data.csv')
    x = np.array(data['km']).reshape(-1, 1)
    y = np.array(data['price']).reshape(-1, 1)
    print (x, y)
    theta = fit_(x, y)
    y_hat = predict_(x, theta)
    plt.scatter(x, y)
    plt.plot(x, y_hat, "r-")
    plt.show()
else:
    print("Usage: ./python3 MyLinearRegression.py")
