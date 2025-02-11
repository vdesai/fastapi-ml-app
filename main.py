from fastapi import FastAPI
import joblib
import numpy as np
from pydantic import BaseModel

# Load the trained model
model = joblib.load("house_price_model.pkl")

# Initialize FastAPI app
app = FastAPI()

# Define request model
class HouseFeatures(BaseModel):
    area: float
    bedrooms: int
    age: float

# Root endpoint
@app.get("/")
def home():
    return {"message": "Welcome to the House Price Prediction API"}

# Prediction endpoint
@app.post("/predict")
def predict_price(features: HouseFeatures):
    data = np.array([[features.area, features.bedrooms, features.age]])
    prediction = model.predict(data)[0]
    return {"predicted_price": round(prediction, 2)}
