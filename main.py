import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Load the dataset
data = pd.read_csv('NairobiOfficePriceEx.csv')
X = data['SIZE'].values
y = data['PRICE'].values


# Function to compute Mean Squared Error
def mean_squared_error(y_true, y_pred):
    return np.mean((y_true - y_pred) ** 2)

# Function for Gradient Descent
def gradient_descent(X, y, m, c, learning_rate, epoch):
    n = len(y)
    errors = []

    for epoch in range(epoch):
        y_pred = m * X + c
        error = mean_squared_error(y, y_pred)
        errors.append(error)

        # Calculate gradients
        dm = (-2 / n) * np.sum(X * (y - y_pred))
        dc = (-2 / n) * np.sum(y - y_pred)

        # Update weights
        m -= learning_rate * dm
        c -= learning_rate * dc

        print(f"Epoch {epoch + 1}: MSE = {error}")
    return m, c, errors


# Initial parameters
m = 0.01
c = -34
learning_rate = 0.0001
epoch = 10

# Train the model
m, c, errors = gradient_descent(X, y, m, c, learning_rate, epoch)

# Plotting the line of best fit
plt.scatter(X, y, color="blue", label="Actual Data")
plt.plot(X, m * X + c, color="red", label="Line of Best Fit")
plt.xlabel("SIZE(sq.ft)")
plt.ylabel("PRICE")
plt.legend()
plt.show()

# Predicting the price for an office size of 100 sq.ft
predicted_price = m * 100 + c
print(f"The predicted office price for a size of 100 sq. ft is: {predicted_price}")
