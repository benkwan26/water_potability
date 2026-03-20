from fastapi import FastAPI
import pandas as pd
import pickle
import os

from data_model import Water

app = FastAPI(
    title="Water Potability Prediction",
    description="Predicts if a water sample is potable using machine learning."
)

with open(os.path.join(os.path.dirname(__file__), '..', 'model.pkl'), 'rb') as f:
    model = pickle.load(f)

@app.get('/')
def index():
    return "Welcome to Water Potability Prediction FastAPI."

@app.post('/predict')
def predict(water: Water):
    sample = pd.DataFrame({
        'ph': [water.ph],
        'Hardness': [water.Hardness],
        'Solids': [water.Solids],
        'Chloramines': [water.Chloramines],
        'Sulfate': [water.Sulfate],
        'Conductivity': [water.Conductivity],
        'Organic_carbon': [water.Organic_carbon],
        'Trihalomethanes': [water.Trihalomethanes],
        'Turbidity': [water.Turbidity]
    })

    prediction = model.predict(sample)

    if prediction == 1:
        return "Water is Potable"
    else:
        return "Water is not Potable"