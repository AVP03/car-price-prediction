import os
import urllib.parse
from django.shortcuts import render
from .forms import PredictionForm
import joblib
import numpy as np
import matplotlib.pyplot as plt
import io
import urllib
import base64
import pandas as pd
from base64 import b64encode
from django.conf import settings

import matplotlib
matplotlib.use('Agg')

def predict_price(request):
    form = PredictionForm(request.POST or None)
    if request.method == 'POST' and form.is_valid():
        engine_power = form.cleaned_data['engine_power']
        model_path = os.path.join(settings.BASE_DIR, 'predictor/car_price_model.pkl')
        model = joblib.load(model_path)
        prediction = model.predict(np.array([[engine_power]]))
        return render(request, 'predictor/result.html', {'prediction': prediction[0]})
    return render(request, 'predictor/predict.html', {'form': form})

def visualize_data(request):
    data = pd.read_csv('/Users/anirudhpentakota/Desktop/car_engine/predictor/mobil_mesin_harga.csv')
    
    
    X = data[['KekuatanMesin']]
    y = data['Harga']

    
    model_path = os.path.join(settings.BASE_DIR, 'predictor/car_price_model.pkl')
    model = joblib.load(model_path)
    
    
    y_pred = model.predict(X)
    
    
    new_engine_power = np.array([[250]])
    new_prediction = model.predict(new_engine_power)

    
    plt.figure(figsize=(10, 6))
    plt.scatter(X, y, color='blue', label='Actual Prices')
    plt.scatter(X, y_pred, color='red', label='Predicted Prices')
    
    
    plt.scatter(new_engine_power, new_prediction, color='green', label='Prediction for 250 HP', s=100, edgecolors='black')

    
    plt.title('Engine Power vs Price')
    plt.xlabel('Engine Power (HP)')
    plt.ylabel('Price ($)')
    plt.legend()
    plt.grid(True)

    
    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    buf.seek(0)
    string = b64encode(buf.read()).decode()
    uri = 'data:image/png;base64,' + urllib.parse.quote(string)

    return render(request, 'predictor/visualization.html', {'data': uri})