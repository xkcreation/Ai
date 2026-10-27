import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

# Independent Variable (House Size)
X = np.array([800, 1000, 1200, 1400, 1600, 1800]).reshape(-1, 1)

# Dependent Variable (House Price)
Y = np.array([35, 45, 55, 65, 75, 85])

# Create the model
model = LinearRegression()

# Train the model
model.fit(X, Y)

# Predict values
predicted = model.predict(X)

# Display model parameters
print("Slope (m):", model.coef_[0])
print("Intercept (c):", model.intercept_)

# Predict the price of a 1525 sq.ft house
new_house = np.array([[1525]])
predicted_price = model.predict(new_house)

print("Predicted Price for 1525 sq.ft house:", predicted_price[0])

# Plot actual data
plt.scatter(X, Y, color="blue", label="Actual Data")

# Plot regression line
plt.plot(X, predicted, color="red", linewidth=2, label="Regression Line")

# Labels
plt.title("House Size vs House Price")
plt.xlabel("House Size (sq.ft)")
plt.ylabel("House Price (Lakhs)")
plt.legend()

# Display graph
plt.show()
