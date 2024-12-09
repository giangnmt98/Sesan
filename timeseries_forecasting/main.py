import argparse
import pandas as pd
from data_utils import load_data, split_data, generate_data, plot_train_test
from arima_model import arima_forecasting
from prophet_model import prophet_forecasting
from autoencoder_model import autoencoder_forecasting
from lstm_model import lstm_forecasting
from evaluate import evaluate_forecast, plot_forecast_comparison

def main():
    parser = argparse.ArgumentParser(description="Chạy mô hình dự báo chuỗi thời gian.")
    parser.add_argument("--file_path", type=str, required=False, help="Đường dẫn tới file CSV.", default="data/data_sample_for_forcasting.csv")
    parser.add_argument("--method", type=str, required=False, choices=["arima", "prophet", "autoencoder", "lstm"], help="Mô hình dự báo.", default="autoencoder")
    parser.add_argument("--split_date", type=str, help="Ngày chia tập dữ liệu (YYYY-MM-DD).")
    args = parser.parse_args()

    # data = load_data(args.file_path)
    data = generate_data(n_samples=1000)

    train, test = split_data(data, split_date=args.split_date)
    plot_train_test(train, test)
    if args.method == "arima":
        forecast = arima_forecasting(train, test)
    elif args.method == "prophet":
        forecast = prophet_forecasting(train, test)
    elif args.method == "autoencoder":
        forecast = autoencoder_forecasting(train, test)
    elif args.method == "lstm":
        forecast = lstm_forecasting(train, test)
    else:
        raise ValueError("Phương pháp không được hỗ trợ.")

    mae, rmse = evaluate_forecast(test['Value'].values, forecast)
    print(f"{args.method.upper()} - MAE: {mae:.2f}, RMSE: {rmse:.2f}")

    plot_forecast_comparison(test, {args.method: forecast})

if __name__ == "__main__":
    main()
