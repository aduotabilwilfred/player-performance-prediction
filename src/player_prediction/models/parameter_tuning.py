import os

import mlflow
import optuna
import pandas as pd
import yaml
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import Ridge

# from xgboost import XGBRegressor


def load_model_type():
    params_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../params.yaml"))
    with open(params_path, "r") as f:
        config = yaml.safe_load(f)
    return config.get("model_type", "random_forest")


train_data = pd.read_csv("data/processed/train.csv")
val_data = pd.read_csv("data/processed/validation.csv")
test_data = pd.read_csv("data/processed/test.csv")

X_train = train_data.drop("player_rating", axis=1)
y_train = train_data["player_rating"]
X_val = val_data.drop("player_rating", axis=1)
y_val = val_data["player_rating"]
X_test = test_data.drop("player_rating", axis=1)
y_test = test_data["player_rating"]


"""X_train = train_data[["consistency_score","market_value_eur","defensive_contribution", "offensive_contribution","creativity_score","goals", "assists"]] #train_df.drop("player_rating", axis=1)
y_train = train_data["player_rating"]
X_val = val_data[["consistency_score","market_value_eur","defensive_contribution", "offensive_contribution","creativity_score","goals", "assists"]] #val_df.drop("player_rating", axis=1)
y_val = val_data["player_rating"]
X_test = test_data[["consistency_score","market_value_eur","defensive_contribution", "offensive_contribution","creativity_score","goals", "assists"]] #test_df.drop("player_rating", axis=1)
y_test = test_data["player_rating"]
"""

mlflow.set_tracking_uri("http://localhost:8000")


def objective_rf(trial, X_train, y_train, X_val, y_val, X_test, y_test):
    params = {
        "n_estimators": trial.suggest_int("n_estimators", 50, 200),
        "max_depth": trial.suggest_int("max_depth", 3, 15),
        "min_samples_split": trial.suggest_int("min_samples_split", 2, 10),
        "random_state": 42,
        "n_jobs": -1,
    }

    with mlflow.start_run(nested=True):
        model = RandomForestRegressor(**params)
        model.fit(X_train, y_train)
        val_score = model.score(X_val, y_val)
        test_score = model.score(X_test, y_test)
        trial.set_user_attr("test_score", test_score)
        return val_score


def objective_gb(trial, X_train, y_train, X_val, y_val, X_test, y_test):
    params = {
        "n_estimators": trial.suggest_int("n_estimators", 50, 200),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.2, log=True),
        "max_depth": trial.suggest_int("max_depth", 3, 8),
        "random_state": 42,
    }

    with mlflow.start_run(nested=True):
        model = GradientBoostingRegressor(**params)
        model.fit(X_train, y_train)
        val_score = model.score(X_val, y_val)
        test_score = model.score(X_test, y_test)
        trial.set_user_attr("test_score", test_score)
        return val_score


def objective_ridge(trial, X_train, y_train, X_val, y_val, X_test, y_test):
    params = {
        "alpha": trial.suggest_float("alpha", 0.01, 10.0, log=True),
        "random_state": 42,
    }

    with mlflow.start_run(nested=True):
        model = Ridge(**params)
        model.fit(X_train, y_train)
        val_score = model.score(X_val, y_val)
        test_score = model.score(X_test, y_test)
        trial.set_user_attr("test_score", test_score)
        return val_score


model_type = load_model_type()

if model_type == "random_forest":
    mlflow.set_experiment("random_forest_parameter_tuning_v2")
    mlflow.sklearn.autolog()
    def objective_func(trial):
        return objective_rf(trial, X_train, y_train, X_val, y_val, X_test, y_test)
elif model_type == "gradient_boosting":
    mlflow.set_experiment("gradient_boosting_parameter_tuning")
    mlflow.sklearn.autolog()
    def objective_func(trial):
        return objective_gb(trial, X_train, y_train, X_val, y_val, X_test, y_test)
elif model_type == "ridge":
    mlflow.set_experiment("ridge_parameter_tuning")
    mlflow.sklearn.autolog()
    def objective_func(trial):
        return objective_ridge(trial, X_train, y_train, X_val, y_val, X_test, y_test)
else:
    raise ValueError(f"Unknown model type for tuning: {model_type}")

with mlflow.start_run():
    study = optuna.create_study(direction="maximize")
    study.optimize(objective_func, n_trials=50)
    mlflow.log_params({f"best_{k}": v for k, v in study.best_params.items()})
    mlflow.log_metric("best_val_score", study.best_value)

    best_test_score = study.best_trial.user_attrs["test_score"]
    mlflow.log_metric("best_test_score", best_test_score)
