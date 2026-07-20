import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import matplotlib.pyplot as plt

# Target variable
y_name = input("Enter target variable name: ")
y = np.array(list(map(float, input(f"Enter {y_name} values (comma-separated): ").split(","))))

# Number of features
n = int(input("Enter number of features: "))

x = {}
names = []

for i in range(n):
    name = input(f"Enter feature {i+1} name: ")
    names.append(name)
    values = list(map(float, input(f"Enter {name} values: ").split(",")))
    x[name] = values

data = pd.DataFrame(x)

# Train model
model = LinearRegression()
model.fit(data, y)

# Prediction on training data
pred = model.predict(data)

# Metrics
mse = mean_squared_error(y, pred)
mae = mean_absolute_error(y, pred)
rmse = np.sqrt(mse)
r2 = r2_score(y, pred)

# Equation
print("\nRegression Equation:")
print(f"{y_name} = {model.intercept_:.2f}", end="")
for i in range(n):
    print(f" + ({model.coef_[i]:.2f} × {names[i]})", end="")
print()

print("\nIntercept:", round(model.intercept_, 2))
for i in range(n):
    print(f"Coefficient of {names[i]}:", round(model.coef_[i], 2))

print("\nMSE :", round(mse, 4))
print("MAE :", round(mae, 4))
print("RMSE:", round(rmse, 4))
print("R2 Score:", round(r2, 4))

# Predict new value
print("\nEnter values for prediction:")
sample = []
for name in names:
    sample.append(float(input(f"{name}: ")))

result = model.predict(pd.DataFrame([sample], columns=names))
print(f"\nPredicted {y_name}: {result[0]:.2f}")

# Graph 1
plt.scatter(y, pred)
plt.plot([min(y), max(y)], [min(y), max(y)], 'r--')
plt.xlabel("Actual")
plt.ylabel("Predicted")
plt.title("Actual vs Predicted")
plt.grid(True)
plt.show()
# Graph 2
metrics = ["R2", "RMSE", "MAE", "MSE"]
values = [r2, rmse, mae, mse]

plt.bar(metrics, values) 
plt.title("Performance Metrics")
plt.ylabel("Value")
plt.grid(axis='y')
plt.show()