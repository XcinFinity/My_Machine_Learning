import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

# 读取数据
Data_50 = pd.read_csv('/Users/lyndonbin/Downloads/VS_Code/data/Machine-Learning/SSE_50_DATA.csv')
Data_50.fillna(0, inplace=True)
Data_50['trade_date'] = pd.to_datetime(Data_50['trade_date'], format='%Y%m%d').dt.strftime('%Y-%m-%d')
# print(Data_50.head())

IDX_Chgsmp_50 = pd.read_csv('/Users/lyndonbin/Downloads/VS_Code/data/Machine-Learning/IDX_Chgsmp.csv')
IDX_Chgsmp_50['Chgsmp01'] = pd.to_datetime(IDX_Chgsmp_50['Chgsmp01'])
IDX_Chgsmp_50['Chgsmp02'] = IDX_Chgsmp_50['Chgsmp02'].astype(str)
IDX_Chgsmp_50['Chgsmp04'] = IDX_Chgsmp_50['Chgsmp04'].astype(int)
# print(IDX_Chgsmp_50.head())

class StockChange:
    def __init__(self, IDX_Chgsmp_50):
        self.IDX_Chgsmp_50 = IDX_Chgsmp_50
        self.Chgsmp_50 = set()
                          
    def update_stocks(self, Time):
        for _, row in self.IDX_Chgsmp_50.iterrows():
            if row['Chgsmp01'] > Time:
                break

            stock_code = row['Chgsmp02']
            change_code = row['Chgsmp04']

            if change_code == 1:
                self.Chgsmp_50.add(stock_code)
            elif change_code == 2:
                self.Chgsmp_50.discard(stock_code)

        print(f"Time: {Time}, Stock: {len(self.Chgsmp_50)}")
        return self.Chgsmp_50

class LSTMModel(nn.Module):
    def __init__(self, input_size=1, hidden_layer_size=100, output_size=1):
        super(LSTMModel, self).__init__()
        self.hidden_layer_size = hidden_layer_size
        self.lstm = nn.LSTM(input_size, hidden_layer_size)
        self.linear = nn.Linear(hidden_layer_size, output_size)
        self.hidden_cell = (torch.zeros(1, 1, self.hidden_layer_size),
                            torch.zeros(1, 1, self.hidden_layer_size))

    def forward(self, input_seq):
        lstm_out, self.hidden_cell = self.lstm(input_seq.view(len(input_seq), 1, -1), self.hidden_cell)
        predictions = self.linear(lstm_out.view(len(input_seq), -1))
        return predictions[-1]

class DatasetBuilder:
    def __init__(self, data, time_step=10):
        self.data = data
        self.time_step = time_step

    def create_dataset(self):
        X, Y = [], []
        for i in range(len(self.data) - self.time_step - 1):
            a = self.data[i:(i + self.time_step), 0]
            X.append(a)
            Y.append(self.data[i + self.time_step, 0])
        return np.array(X), np.array(Y)

    def build_datasets(self, train_data, test_data):
        # 预处理数据
        scaler = MinMaxScaler(feature_range=(0, 1))
        train_scaled = scaler.fit_transform(train_data[['收盘价']])  # 请将'收盘价'替换为实际的列名
        test_scaled = scaler.transform(test_data[['收盘价']])

        # 创建数据集
        X_train, Y_train = self.create_dataset(train_scaled)
        X_test, Y_test = self.create_dataset(test_scaled)

        # 转换为张量
        X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], 1)
        X_test = X_test.reshape(X_test.shape[0], X_test.shape[1], 1)
        X_train = torch.from_numpy(X_train).float()
        Y_train = torch.from_numpy(Y_train).float()
        X_test = torch.from_numpy(X_test).float()
        Y_test = torch.from_numpy(Y_test).float()

        # 创建数据加载器
        train_dataset = TensorDataset(X_train, Y_train)
        test_dataset = TensorDataset(X_test, Y_test)
        train_loader = DataLoader(dataset=train_dataset, batch_size=64, shuffle=True)
        test_loader = DataLoader(dataset=test_dataset, batch_size=64, shuffle=False)

        return train_loader, test_loader, scaler

# 读取数据
Data_50 = pd.read_csv('/Users/lyndonbin/Downloads/VS_Code/data/Machine-Learning/SSE_50_DATA.csv')
Data_50.fillna(0, inplace=True)
Data_50['trade_date'] = pd.to_datetime(Data_50['trade_date'], format='%Y%m%d').dt.strftime('%Y-%m-%d')
# print(Data_50.head())

IDX_Chgsmp_50 = pd.read_csv('/Users/lyndonbin/Downloads/VS_Code/data/Machine-Learning/IDX_Chgsmp.csv')
IDX_Chgsmp_50['Chgsmp01'] = pd.to_datetime(IDX_Chgsmp_50['Chgsmp01'])
IDX_Chgsmp_50['Chgsmp02'] = IDX_Chgsmp_50['Chgsmp02'].astype(str)
IDX_Chgsmp_50['Chgsmp04'] = IDX_Chgsmp_50['Chgsmp04'].astype(int)
# print(IDX_Chgsmp_50.head())

stock_change = StockChange(IDX_Chgsmp_50)

# 实现时间滚动
start_date = pd.Timestamp('2000-01-01')
end_date = pd.Timestamp('2024-01-01')
current_date = start_date

while current_date <= end_date:
    current_stocks = stock_change.update_stocks(current_date)
    
    for stock in current_stocks:
        stock_data = Data_50[Data_50['ts_code'] == stock]  # 请将'股票代码'替换为实际的股票代码列名
        stock_data = stock_data[stock_data['trade_date'] <= current_date]
        
        # 获取前三年的数据作为训练集
        train_data = stock_data[(stock_data['trade_date'] > current_date - pd.DateOffset(years=3)) & (stock_data['trade_date'] <= current_date)]
        
        # 获取后一周的数据作为测试集
        test_data = stock_data[(stock_data['trade_date'] > current_date) & (stock_data['trade_date'] <= current_date + pd.DateOffset(weeks=1))]
        
        if len(train_data) > 0 and len(test_data) > 0:
            dataset_builder = DatasetBuilder(data=Data_50)
            train_loader, test_loader, scaler = dataset_builder.build_datasets(train_data, test_data)

            # 构建 LSTM 模型
            model = LSTMModel()
            loss_function = nn.MSELoss()
            optimizer = optim.Adam(model.parameters(), lr=0.001)

            # 训练模型
            epochs = 100
            for epoch in range(epochs):
                for seq, labels in train_loader:
                    optimizer.zero_grad()
                    model.hidden_cell = (torch.zeros(1, 1, model.hidden_layer_size),
                                         torch.zeros(1, 1, model.hidden_layer_size))

                    y_pred = model(seq)

                    single_loss = loss_function(y_pred, labels)
                    single_loss.backward()
                    optimizer.step()

                if epoch % 10 == 0:
                    print(f'Epoch {epoch} loss: {single_loss.item()}')

            # 测试模型
            model.eval()
            test_predictions = []
            with torch.no_grad():
                for seq, labels in test_loader:
                    model.hidden_cell = (torch.zeros(1, 1, model.hidden_layer_size),
                                         torch.zeros(1, 1, model.hidden_layer_size))
                    y_pred = model(seq)
                    test_predictions.append(y_pred.numpy())

            test_predictions = np.concatenate(test_predictions)
            test_predictions = scaler.inverse_transform(test_predictions.reshape(-1, 1))
            print(f"Stock: {stock}, Predictions: {test_predictions}")

    # 时间滚动
    current_date += pd.DateOffset(months=1)