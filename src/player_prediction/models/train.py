import json
import os
import pickle
import numpy as np
import pandas as pd
import yaml
from sklearn.metrics import mean_absolute_error, mean_squared_error
from xgboost import XGBRegressor


def load_parameters():
    params_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../params.yaml"))
    with open(params_path, "r") as f:
        config = yaml.safe_load(f)
    return config["xgboost_train"]


def main(X_train, y_train, X_val, y_val, X_test, y_test, parameters=None):
    params = parameters
    print(params)

    model = XGBRegressor(
        n_estimators=params["n_estimators"],
        max_depth=params["max_depth"],
        learning_rate=params["learning_rate"],
        random_state=params["random_state"],
        n_jobs=params["n_jobs"],
    )

    model.fit(X_train, y_train)
    val_r2 = model.score(X_val, y_val)
    test_r2 = model.score(X_test, y_test)
    val_pred = model.predict(X_val)
    val_rmse = np.sqrt(mean_squared_error(y_val, val_pred))
    val_mae = mean_absolute_error(y_val, val_pred)
    test_pred = model.predict(X_test)
    test_rmse = np.sqrt(mean_squared_error(y_test, test_pred))
    test_mae = mean_absolute_error(y_test, test_pred)

    metrics = {
        "val_r2": f"{val_r2:.2f}",
        "test_r2": f"{test_r2:.2f}",
        "val_rmse": f"{val_rmse:.2f}",
        "val_mae": f"{val_mae:.2f}",
        "test_rmse": f"{test_rmse:.2f}",
        "test_mae": f"{test_mae:.2f}",
    }
    with open("metrics.json", "w") as f:
        json.dump(metrics, f)

    model_path = os.path.join(os.path.dirname(__file__), "../../../models/model.pkl")
    with open(os.path.abspath(model_path), "wb") as f:
        pickle.dump(model, f)


if __name__ == "__main__":
    # load data
    train_df = pd.read_csv("data/processed/train.csv")
    val_df = pd.read_csv("data/processed/validation.csv")
    test_df = pd.read_csv("data/processed/test.csv")

    # separate features and target
    X_train = train_df.drop("player_rating", axis=1)
    y_train = train_df["player_rating"]
    X_val = val_df.drop("player_rating", axis=1)
    y_val = val_df["player_rating"]
    X_test = test_df.drop("player_rating", axis=1)
    y_test = test_df["player_rating"]
    main(X_train, y_train, X_val, y_val, X_test, y_test, parameters=load_parameters())
