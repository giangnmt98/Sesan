import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
import numpy as np
import optuna
from evaluate import evaluate_forecast

class Autoencoder(nn.Module):
    def __init__(self, input_dim, encoding_dim):
        super(Autoencoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, encoding_dim),
            nn.ReLU(),
            nn.Linear(encoding_dim, encoding_dim // 2),
            nn.ReLU(),
            nn.Linear(encoding_dim // 2, encoding_dim // 4),
            nn.ReLU()
        )
        self.decoder = nn.Sequential(
            nn.Linear(encoding_dim // 4, encoding_dim // 2),
            nn.ReLU(),
            nn.Linear(encoding_dim // 2, encoding_dim),
            nn.ReLU(),
            nn.Linear(encoding_dim, input_dim),
            nn.Sigmoid()
        )

    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded

def autoencoder_forecasting(train, test, encoding_dim=100, epochs=2000, batch_size=16, learning_rate=0.0001):
    """
    Dự báo dữ liệu sử dụng Autoencoder.
    Args:
        train (pd.DataFrame): Tập dữ liệu train
        test (pd.DataFrame): Tập dữ liệu test
        encoding_dim (int): Số chiều của lớp mã hóa
        epochs (int): Số epoch huấn luyện
        batch_size (int): Kích thước batch
        learning_rate (float): Tốc độ học
    Returns:
        np.ndarray: Giá trị dự báo của Autoencoder
    """
    # Chuẩn hóa dữ liệu
    train_values = train['Value'].values
    test_values = test['Value'].values
    scaler = (train_values.min(), train_values.max())
    train_scaled = (train_values - scaler[0]) / (scaler[1] - scaler[0])
    test_scaled = (test_values - scaler[0]) / (scaler[1] - scaler[0])
    # train_scaled = train_values
    # test_scaled = test_values

    # Sử dụng kích thước ngắn nhất cho Autoencoder
    input_dim = min(len(train_scaled), len(test_scaled))
    train_scaled = train_scaled[:input_dim]
    test_scaled = test_scaled[:input_dim]

    # Chuyển dữ liệu sang Tensor
    train_tensor = torch.tensor(train_scaled, dtype=torch.float32).unsqueeze(0)

    # Khởi tạo Autoencoder
    model = Autoencoder(input_dim=input_dim, encoding_dim=encoding_dim)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # Huấn luyện Autoencoder với tqdm
    train_losses = []
    model.train()
    for epoch in tqdm(range(epochs), desc="Training Autoencoder"):
        optimizer.zero_grad()
        output = model(train_tensor)
        loss = criterion(output, train_tensor)
        loss.backward()
        optimizer.step()
        train_losses.append(loss.item())

    # Dự báo
    test_tensor = torch.tensor(test_scaled, dtype=torch.float32).unsqueeze(0)
    model.eval()
    with torch.no_grad():
        reconstructed = model(test_tensor).squeeze(0).numpy()

    forecast = reconstructed * (scaler[1] - scaler[0]) + scaler[0]
    return forecast, encoding_dim, epochs, learning_rate
