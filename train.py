import sqlite3
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

import common


def load_train_data(path):
    print(f"Reading train data from the database: {path}")
    con = sqlite3.connect(path)
    data_train = pd.read_sql("SELECT * FROM train", con)
    con.close()
    X = data_train.drop(columns=["target"])
    y = data_train["target"]
    return X, y


def fit_model(X, y):
    print(f"Fitting a model")
    model = LinearRegression()
    model.fit(X, y)
    y_pred = model.predict(X)
    score = mean_squared_error(y, y_pred)
    print(f"Score on train data {score:.2f}")
    return model


if __name__ == "__main__":
    X_train, y_train = load_train_data(common.DB_PATH)
    X_train = common.preprocess_data(X_train)
    model = fit_model(X_train, y_train)
    common.persist_model(model, common.MODEL_PATH)
