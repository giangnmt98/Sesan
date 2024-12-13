import optuna
from evaluate import evaluate_forecast
from autoencoder_model import autoencoder_forecasting

def hyperparameter_optimization(train, test, n_trials=100):
    def objective(trial):
        # Suggest hyperparameters
        encoding_dim = trial.suggest_int("encoding_dim", 100, 1000)
        epochs = trial.suggest_int("epochs", 100, 5000)
        learning_rate = trial.suggest_float("learning_rate", 0.0001, 0.1)

        forecast,_,_,_ = autoencoder_forecasting(train, test, encoding_dim=encoding_dim, epochs=epochs,
        batch_size=learning_rate)
        mae, rmse = evaluate_forecast(test['Value'].values, forecast)
        return rmse
    # Create an Optuna study and optimize
    study = optuna.create_study(direction="minimize")
    study.optimize(objective, n_trials=n_trials)

    # Print best parameters
    print("Best parameters:", study.best_params)
    print("Best RMSE:", study.best_value)
    return study.best_params
