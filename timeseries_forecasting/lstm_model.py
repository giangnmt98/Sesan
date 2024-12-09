import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
import numpy as np

class LSTM(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers, output_dim, dropout=0.2):
        super(LSTM, self).__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True, dropout=dropout)
        self.fc1 = nn.Linear(hidden_dim, hidden_dim // 2)
        self.fc2 = nn.Linear(hidden_dim // 2, output_dim)
        self.relu = nn.ReLU()

    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        x = self.relu(self.fc1(lstm_out[:, -1, :]))
        output = self.fc2(x)
        return output

def lstm_forecasting(train, test, hidden_dim=50, num_layers=1, epochs=1000, learning_rate=0.001):
    """
    Dự báo dữ liệu sử dụng LSTM.
    Args:
        train (pd.DataFrame): Tập dữ liệu train
        test (pd.DataFrame): Tập dữ liệu test
        hidden_dim (int): Số chiều của lớp ẩn LSTM
        num_layers (int): Số lớp LSTM
        epochs (int): Số epoch huấn luyện
        learning_rate (float): Tốc độ học
    Returns:
        np.ndarray: Giá trị dự báo của LSTM
    """
    # Chuẩn hóa dữ liệu
    train_values = train['Value'].values
    test_values = test['Value'].values
    scaler = (train_values.min(), train_values.max())
    train_scaled = (train_values - scaler[0]) / (scaler[1] - scaler[0])
    test_scaled = (test_values - scaler[0]) / (scaler[1] - scaler[0])

    # Chuyển đổi dữ liệu thành dạng chuỗi thời gian (sequence)
    def create_sequences(data, seq_length):
        sequences = []
        for i in range(len(data) - seq_length):
            seq = data[i:i + seq_length]
            label = data[i + seq_length]
            sequences.append((seq, label))
        return sequences

    seq_length = 10
    train_sequences = create_sequences(train_scaled, seq_length)

    # Chuyển đổi thành Tensor
    train_x = torch.tensor([s[0] for s in train_sequences], dtype=torch.float32).unsqueeze(-1)
    train_y = torch.tensor([s[1] for s in train_sequences], dtype=torch.float32)

    # Khởi tạo mô hình LSTM
    input_dim = 1
    output_dim = 1
    model = LSTM(input_dim, hidden_dim, num_layers, output_dim)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # Huấn luyện LSTM với tqdm
    model.train()
    for epoch in tqdm(range(epochs), desc="Training LSTM"):
        optimizer.zero_grad()
        output = model(train_x)
        loss = criterion(output.squeeze(), train_y)
        loss.backward()
        optimizer.step()

    # Dự báo
    model.eval()
    test_inputs = train_scaled[-seq_length:].tolist()
    predictions = []

    for _ in range(len(test_scaled)):
        seq = torch.tensor(test_inputs[-seq_length:], dtype=torch.float32).unsqueeze(0).unsqueeze(-1)
        with torch.no_grad():
            pred = model(seq)
            test_inputs.append(pred.item())
            predictions.append(pred.item())

    # Chuyển đổi giá trị dự báo về thang đo ban đầu
    forecast = np.array(predictions) * (scaler[1] - scaler[0]) + scaler[0]
    return forecast
