import pandas as pd
import pickle
import os
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

def run():
    data = {
        'Horas_Estudio': [1,2,3,4,5,6,7,8,9,10],
        'Calificacion':  [50,55,60,65,70,75,80,85,90,95]
    }
    df = pd.DataFrame(data)
    X = df[['Horas_Estudio']]
    y = df['Calificacion']

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42
    )
    model = LinearRegression()
    model.fit(X_train, y_train)

    os.makedirs("models", exist_ok=True)
    with open("models/model.pkl", "wb") as f:
        pickle.dump(model, f)
    print("Modelo guardado en models/model.pkl")

if __name__ == "__main__":
    run()

