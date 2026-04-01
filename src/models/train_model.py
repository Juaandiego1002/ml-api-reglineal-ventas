import os
import joblib
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

def train_model(X, y):
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    model = LinearRegression()
    model.fit(X_train, y_train)
    return model, X_test, y_test

if __name__ == "__main__":
    data = {
        'Horas_Estudio': [1,2,3,4,5,6,7,8,9,10],
        'Calificacion':  [50,55,60,65,70,75,80,85,90,95]
    }
    df = pd.DataFrame(data)
    X = df[['Horas_Estudio']]
    y = df['Calificacion']

    model, X_test, y_test = train_model(X, y)

    os.makedirs("models", exist_ok=True)
    joblib.dump(model, "models/model.pkl")
    print("✅ Modelo guardado en models/model.pkl")