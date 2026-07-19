import json
import os
import pickle

import numpy as np
import pandas as pd
import yaml
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error
from xgboost import XGBRegressor


def load_parameters():
    params_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../params.yaml"))
    with open(params_path, "r") as f:
        config = yaml.safe_load(f)
    model_type = config.get("model_type", "random_forest")
    return model_type, config[model_type]


def get_model(model_type, params):
    if model_type == "random_forest":
        return RandomForestRegressor(**params)
    elif model_type == "gradient_boosting":
        return GradientBoostingRegressor(**params)
    elif model_type == "ridge":
        return Ridge(**params)
    elif model_type == "xgboost":
        return XGBRegressor(**params)
    else:
        raise ValueError(f"Unknown model type: {model_type}")


def main(X_train, y_train, X_val, y_val, X_test, y_test, model_type, parameters=None):
    params = parameters
    print(f"Training {model_type} with parameters: {params}")

    model = get_model(model_type, params)

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
    X_train = train_df[
        [
            "consistency_score",
            "market_value_eur",
            "defensive_contribution",
            "offensive_contribution",
            "creativity_score",
            "goals",
            "assists",
        ]
    ]  # train_df.drop("player_rating", axis=1)
    y_train = train_df["player_rating"]
    X_val = val_df[
        [
            "consistency_score",
            "market_value_eur",
            "defensive_contribution",
            "offensive_contribution",
            "creativity_score",
            "goals",
            "assists",
        ]
    ]  # val_df.drop("player_rating", axis=1)
    y_val = val_df["player_rating"]
    X_test = test_df[
        [
            "consistency_score",
            "market_value_eur",
            "defensive_contribution",
            "offensive_contribution",
            "creativity_score",
            "goals",
            "assists",
        ]
    ]  # test_df.drop("player_rating", axis=1)
    y_test = test_df["player_rating"]
    model_type, parameters = load_parameters()
    main(X_train, y_train, X_val, y_val, X_test, y_test, model_type=model_type, parameters=parameters)
