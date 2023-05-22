import pandas as pd
from sklearn.linear_model import LinearRegression

# Load the data from CSV
data = pd.read_csv('data.csv')

# Extract the features (km) and target variable (price)
x = data['km'].values.reshape(-1, 1)
y = data['price'].values

# Initialize and fit the linear regression model
model = LinearRegression()
model.fit(x, y)
print()

# Predict the prices for the given mileage values
predictions = model.predict(x)

# Print the coefficients (theta values) and the intercept (theta_0)
print("Coefficients: ", model.coef_)
print("Intercept: ", model.intercept_)

# Plot the regression line and the data points
import matplotlib.pyplot as plt
plt.scatter(x, y)
plt.plot(x, predictions, color='red')
plt.xlabel('Mileage (km)')
plt.ylabel('Price')
plt.savefig('image2.png')
