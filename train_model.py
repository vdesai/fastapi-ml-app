import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error
import joblib

# Generate a simple dataset (House Prices Example)
np.random.seed(42)
num_samples = 100
area = np.random.randint(500, 5000, num_samples)  # House area in sqft
bedrooms = np.random.randint(1, 5, num_samples)  # Number of bedrooms
age = np.random.randint(1, 50, num_samples)  # Age of the house
price = area * 100 + bedrooms * 5000 - age * 200 + np.random.randint(10000, 50000, num_samples)  # House price formula

# Create a DataFrame
df = pd.DataFrame({"Area": area, "Bedrooms": bedrooms, "Age": age, "Price": price})

# Split into training and test sets
X = df[["Area", "Bedrooms", "Age"]]
y = df["Price"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train a simple Linear Regression model
model = LinearRegression()
model.fit(X_train, y_train)

# Make predictions and evaluate
y_pred = model.predict(X_test)
mae = mean_absolute_error(y_test, y_pred)
print(f"Model trained. Mean Absolute Error: {mae:.2f}")

# Save the trained model
joblib.dump(model, "house_price_model.pkl")
print("Model saved as house_price_model.pkl")
