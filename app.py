from fastapi import FastAPI
from pydantic import BaseModel
import pickle
import numpy as np

with open('gradient_boosting_model.pkl', 'rb') as file:
    model = pickle.load(file)

# Define FastAPI app
app = FastAPI()

# Define a Pydantic model for the input data
class InputData(BaseModel):
    A6: float
    A9: float
    A24: float
    A27: float
    A29: float
    A34: float
    A35: float
    A39: float
    A40: float
    A44: float
    A46: float
    A56: float
    A58: float
    A61: float

# Define a prediction endpoint
@app.post('/predict')
def predict(input_data: InputData):
    # Convert input data to a DataFrame or array
    data = np.array([[input_data.A6, input_data.A9, input_data.A24, input_data.A27,
                      input_data.A29, input_data.A34, input_data.A35, input_data.A39,
                      input_data.A40, input_data.A44, input_data.A46, input_data.A56,
                      input_data.A58, input_data.A61]])

    # Make prediction
    prediction = model.predict(data)
    
    return {'prediction': prediction.tolist()}  # Convert to list for JSON response

# Optionally, define a root endpoint
@app.get('/')
def read_root():
    return {'message': 'Welcome to the Bankruptcy Prediction API'}
