import joblib
import numpy as np

model = joblib.load("models/modelo_regresion.pkl")

def predict(publicidad):
    data = np.array([[publicidad]])
    pred = model.predict(data)
    return float(pred[0])