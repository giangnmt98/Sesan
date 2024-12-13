import argparse
import pandas as pd
from data_utils import load_data, split_data, generate_data, plot_train_test
from arima_model import arima_forecasting
from prophet_model import prophet_forecasting
from autoencoder_model import autoencoder_forecasting
from lstm_model import lstm_forecasting
from evaluate import evaluate_forecast, plot_forecast_comparison
from hyperparameter_optimization import hyperparameter_optimization


def ghi_du_lieu(data):
    file_path = "experiment.txt"

    try:
        # Ghi dữ liệu vào file
        with open(file_path, "a", encoding="utf-8") as file:
            file.write(data + "\n")
        print(f"Dữ liệu đã được ghi vào {file_path}.")
    except Exception as e:
        # Xử lý khi có lỗi xảy ra
        print(f"Đã xảy ra lỗi: {e}")


def main():
    parser = argparse.ArgumentParser(description="Chạy mô hình dự báo chuỗi thời gian.")
    parser.add_argument("--file_path", type=str, required=False, help="Đường dẫn tới file CSV.", default="data/data_luu_luong_processed.csv")
    parser.add_argument("--method", type=str, required=False, choices=["arima", "prophet", "autoencoder", "lstm"], help="Mô hình dự báo.", default="autoencoder")
    parser.add_argument("--split_date", type=str, help="Ngày chia tập dữ liệu (YYYY-MM-DD).", default="2024-11-30")
    args = parser.parse_args()

    data = load_data(args.file_path)
    data =  data[data['Value'] >= 0]

    train, test = split_data(data, split_date=args.split_date)
    plot_train_test(train, test)
    if args.method == "arima":
        forecast = arima_forecasting(train, test)
    elif args.method == "prophet":
        forecast = prophet_forecasting(train, test)
    elif args.method == "autoencoder":
        forecast, encoding_dim, epochs, learning_rate  = autoencoder_forecasting(train, test)
    elif args.method == "lstm":
        forecast = lstm_forecasting(train, test)
    else:
        raise ValueError("Phương pháp không được hỗ trợ.")
    plot_forecast_comparison(test, {args.method: forecast})
    # mae, rmse = evaluate_forecast(test['Value'].values, forecast)
    # experiment_text = f"{args.method.upper()} - Encoding_dim: {encoding_dim}, Epochs: {epochs}, Learning Rate:  {learning_rate}, MAE: {mae:.2f}, RMSE: {rmse:.2f}"
    # print(experiment_text)

    # optimizer_hyperparameter = hyperparameter_optimization(train,test, n_trials=50)
    # encoding_dim = optimizer_hyperparameter['encoding_dim']
    # epochs = optimizer_hyperparameter['epochs']
    # learning_rate = optimizer_hyperparameter['learning_rate']

    # forecast, encoding_dim, epochs, learning_rate = autoencoder_forecasting(train, test,encoding_dim=283,
    #                                                                         epochs=133, learning_rate= 0.04290051354457626 )
    # forecast, encoding_dim, epochs, learning_rate = autoencoder_forecasting(train, test)
    # plot_forecast_comparison(test, {args.method: forecast})

if __name__ == "__main__":
    main()
