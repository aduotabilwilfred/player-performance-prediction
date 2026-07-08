import mlflow
import optuna
import pandas as pd
from xgboost import XGBRegressor

train_data = pd.read_csv("data/processed/train.csv")
val_data = pd.read_csv("data/processed/validation.csv")
test_data = pd.read_csv("data/processed/test.csv")

X_train = train_data.drop("player_rating", axis=1)
y_train = train_data["player_rating"]
X_val = val_data.drop("player_rating", axis=1)
y_val = val_data["player_rating"]
X_test = test_data.drop("player_rating", axis=1)
y_test = test_data["player_rating"]


mlflow.set_tracking_uri("http://localhost:8000")
mlflow.set_experiment("xgboost_parameter_tuning_v2")
mlflow.xgboost.autolog()


def objective(trial, X_train, y_train, X_val, y_val, X_test, y_test):
    params = {
        "n_estimators": trial.suggest_int("n_estimators", 50, 300),
        "max_depth": trial.suggest_int("max_depth", 3, 10),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
        "subsample": trial.suggest_float("subsample", 0.6, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
        "n_jobs": trial.suggest_int("n_jobs", -1, -1),
    }

    with mlflow.start_run(nested=True):
        model = XGBRegressor(**params, random_state=42)
        model.fit(X_train, y_train)
        val_score = model.score(X_val, y_val)
        test_score = model.score(X_test, y_test)
        trial.set_user_attr("test_score", test_score)
        return val_score


with mlflow.start_run():
    study = optuna.create_study(direction="maximize")
    study.optimize(lambda trial: objective(trial, X_train, y_train, X_val, y_val, X_test, y_test), n_trials=50)
    mlflow.log_params({f"best_{k}": v for k, v in study.best_params.items()})
    mlflow.log_metric("best_val_score", study.best_value)

    best_test_score = study.best_trial.user_attrs["test_score"]
    mlflow.log_metric("best_test_score", best_test_score)
