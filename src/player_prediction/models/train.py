import json
import os
import pickle

import pandas as pd
import yaml
from xgboost import XGBRegressor


def load_parameters():
    params_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../params.yaml"))
    with open(params_path, "r") as f:
        config = yaml.safe_load(f)
    return config["xgboost_train"]


def main():
    params = load_parameters()
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
    metrics = {"test_r2": test_r2, "val_r2": val_r2}
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
    main()
