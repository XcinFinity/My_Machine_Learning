import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_percentage_error, mean_squared_error, mean_absolute_error, r2_score, mean_squared_log_error
from tqdm import tqdm
import pickle

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset


all_data = pd.read_csv('A500_DATA.csv')

all_datas = all_data.groupby('ts_code').apply(lambda x: x.sort_values('trade_date')).reset_index(drop = True)
all_datas['trade_date'] = pd.to_datetime(all_datas['trade_date'], format = '%Y%m%d')

all_datas.fillna(0, inplace=True)
print(all_datas)


all_datas.columns

all_datas.duplicated().sum()

all_datas.isnull().sum().sum()


stock_change = set(all_datas['ts_code'].unique())

stock_feature = all_datas[[ ]]


class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size, dropout=0.5):
        super(LSTMModel, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, dropout=dropout)
        self.fc = nn.Linear(hidden_size, output_size)
    
    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        out, _ = self.lstm(x, (h0, c0))
        out = self.fc(out[:, -1, :])
        return out

def prepare_data(start_index, end_index, window_size, prediction_size, scaled_features, all_datas):
    X_train = []
    y_train = []
    for i in range(start_index, end_index):
        if i < window_size or i + prediction_size > len(scaled_features):
            continue
        X_train.append(scaled_features[i-window_size:i])
        y_train.append(all_datas['close'].values[i:i+prediction_size])
    X_train = np.array(X_train)
    y_train = np.array([y for y in y_train if len(y) == prediction_size])
    valid_indices = [i for i in range(len(y_train)) if len(y_train[i]) == prediction_size]
    X_train, y_train = X_train[valid_indices], y_train[valid_indices]
    return X_train, y_train

def train_model(model, train_loader, criterion, optimizer, device, current_date, stock_code):
    model.train()
    print(f'Starting training for {stock_code} at {current_date.date()}')
    for epoch in tqdm(range(200), desc=f'Training {stock_code}'):  # 每次训练200个epoch
        running_loss = 0.0
        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            optimizer.zero_grad()
            loss = criterion(model(X_batch), y_batch)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
    print(f'[{current_date.date()}] Training completed for {stock_code}, Final Loss: {running_loss / len(train_loader):.4f}')


def predict(model, scaled_features, end_index, window_size, input_size, device):
    model.eval()
    X_test = torch.tensor(scaled_features[end_index-window_size:end_index].reshape(1, window_size, input_size), dtype=torch.float32).to(device)
    with torch.no_grad():
        return model(X_test).cpu().numpy().flatten()


# 初始化参数
start_date = pd.to_datetime('2004-01-03')
end_date = pd.to_datetime('2024-01-01')
current_date = start_date
window_size = 252 * 3
prediction_size = 5

hidden_size = 100
num_layers = 3
output_size = prediction_size
dropout = 0.5

features = stock_feature
scaled_features = MinMaxScaler().fit_transform(features)
input_size = scaled_features.shape[1]

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = LSTMModel(input_size, hidden_size, num_layers, output_size, dropout)
if torch.cuda.device_count() > 1:
    print(f"Using {torch.cuda.device_count()} GPUs!")
    model = nn.DataParallel(model)
model.to(device)

