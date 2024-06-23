import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import joblib
import os

# Load the dataset
data = pd.read_csv('/Users/anirudhpentakota/Desktop/car_engine/car_price_prediction/predictor/mobil_mesin_harga.csv')

# Define the feature and target variable
X = data[['KekuatanMesin']]
y = data['Harga']

# Split the data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create and train the model
model = LinearRegression()
model.fit(X_train, y_train)

# Ensure the directory exists
model_dir = '/Users/anirudhpentakota/Desktop/car_engine/car_price_prediction/predictor'
if not os.path.exists(model_dir):
    os.makedirs(model_dir)

# Save the trained model
model_path = os.path.join(model_dir, 'car_price_model.pkl')
joblib.dump(model, model_path)

print(f"Model has been trained and saved as {model_path}")