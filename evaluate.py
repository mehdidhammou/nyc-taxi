import sqlite3
import pandas as pd
from sklearn.metrics import mean_squared_error

import common

def load_test_data(path):
    print(f"Reading test data from the database: {path}")
    con = sqlite3.connect(path)
    data_test = pd.read_sql('SELECT * FROM test', con)
    con.close()
    X = data_test.drop(columns=['target'])
    y = data_test['target']
    return X, y

def evaluate_model(model, X, y):
    print(f"Evaluating the model")
    y_pred = model.predict(X)
    score = mean_squared_error(y, y_pred)
    return score

if __name__ == "__main__":

    X_test, y_test = load_test_data(common.DB_PATH)
    X_test = common.preprocess_data(X_test)
    model = common.load_model(common.MODEL_PATH)
    score_test = evaluate_model(model, X_test, y_test)
    print(f"Score on test data {score_test:.2f}")
