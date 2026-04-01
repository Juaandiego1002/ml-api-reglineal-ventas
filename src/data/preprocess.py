def prepare_data(df):
    X = df[["Publicidad"]]
    y = df["Venta"]
    return X, y