import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_percentage_error, mean_squared_error, mean_absolute_error, r2_score, mean_squared_log_error
from tqdm import tqdm
import pickle
import os

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from PIL import Image

# 读取指数成分变更数据
IDX_Chgsmp_50 = pd.read_csv('IDX_Chgsmp.csv')

IDX_Chgsmp_50['Chgsmp01'] = pd.to_datetime(IDX_Chgsmp_50['Chgsmp01'])
IDX_Chgsmp_50['Chgsmp02'] = IDX_Chgsmp_50['Chgsmp02'].apply(lambda x: str(x) + '.SH')
IDX_Chgsmp_50['Chgsmp04'] = IDX_Chgsmp_50['Chgsmp04'].astype(int)
print(IDX_Chgsmp_50)

# 读取股票数据
all_data = pd.read_csv('SSE_50_DATA.csv')

all_datas = all_data.groupby('ts_code').apply(lambda x: x.sort_values('trade_date')).reset_index(drop=True)
all_datas['trade_date'] = pd.to_datetime(all_datas['trade_date'], format='%Y%m%d')

# all_datas.fillna(np.nan, inplace=True)
all_datas.fillna(0, inplace=True)
# all_datas.interpolate(method='linear', inplace=True)
print(all_datas)

# 初始化股票代码列表
stock_change = set(all_datas['ts_code'].unique())

# 定义数据转换为图像的函数
def data_to_image(data, save_path):
    plt.figure(figsize=(10, 6))
    plt.plot(data)
    plt.savefig(save_path)
    plt.close()

# 定义CNN模型
class CNNModel(nn.Module):
    def __init__(self, num_classes):
        super(CNNModel, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.fc1 = nn.Linear(64 * 8 * 8, 128)
        self.fc2 = nn.Linear(128, num_classes)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 64 * 8 * 8)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

# 定义数据集类
class ImageDataset(Dataset):
    def __init__(self, image_paths, labels, transform=None):
        self.image_paths = image_paths
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image = Image.open(self.image_paths[idx])
        if self.transform:
            image = self.transform(image)
        label = self.labels[idx]
        return image, label

# 初始化参数
start_date = pd.to_datetime('2004-01-03')
end_date = pd.to_datetime('2024-01-01')
current_date = start_date
window_size = 252 * 3
prediction_size = 5

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = CNNModel(num_classes=prediction_size)
if torch.cuda.device_count() > 1:
    print(f"Using {torch.cuda.device_count()} GPUs!")
    model = nn.DataParallel(model)
model.to(device)

# 保存所有实际值和预测值
all_actual_values = []
all_predictions = []
all_dates = []

# 保存每次 current_date 的股票代码和预测数据
results = {}
metrics = []

# 初始化交易系统
initial_cash = 10000
cash = initial_cash
portfolio_value = []
portfolio_dates = []

scaler = MinMaxScaler()
scaled_features = scaler.fit_transform(stock_feature)
criterion = nn.MSELoss()

transform = transforms.Compose([
    transforms.Resize((32, 32)),
    transforms.ToTensor(),
])

while current_date < end_date:
    # 根据成份股变动数据调整股票代码列表
    changes = IDX_Chgsmp_50[IDX_Chgsmp_50['Chgsmp01'] <= current_date]
    for _, row in changes.iterrows():
        if row['Chgsmp04'] == 1:
            stock_change.add(row['Chgsmp02'])
        elif row['Chgsmp04'] == 2:
            stock_change.discard(row['Chgsmp02'])

    current_results = {}
    stock_returns = []

    for stock_code in tqdm(stock_change, desc=f'Processing {current_date.date()}'):
        stock_data = all_datas[all_datas['ts_code'] == stock_code]
        if len(stock_data) < window_size + prediction_size:
            continue

        start_index = stock_data[stock_data['trade_date'] >= current_date].index
        if len(start_index) == 0:
            continue
        start_index = start_index[0]
        end_index = start_index + window_size
        if end_index + prediction_size > len(stock_data):
            continue

        # 将数据转换为图像并保存
        image_path = f'images/{stock_code}_{current_date.date()}.png'
        os.makedirs(os.path.dirname(image_path), exist_ok=True)
        data_to_image(stock_data['close'].values[start_index:end_index], image_path)

        # 创建数据集和数据加载器
        image_paths = [image_path]
        labels = [stock_data['close'].values[end_index:end_index+prediction_size]]
        dataset = ImageDataset(image_paths, labels, transform=transform)
        dataloader = DataLoader(dataset, batch_size=1, shuffle=True)

        optimizer = optim.Adam(model.parameters(), lr=0.001)

        # 训练模型
        model.train()
        for epoch in range(200):
            running_loss = 0.0
            for images, labels in dataloader:
                images, labels = images.to(device), labels.to(device)
                optimizer.zero_grad()
                outputs = model(images)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
                running_loss += loss.item()
        print(f'[{current_date.date()}] Training completed for {stock_code}, Final Loss: {running_loss / len(dataloader):.4f}')

        # 预测
        model.eval()
        with torch.no_grad():
            for images, _ in dataloader:
                images = images.to(device)
                prediction = model(images).cpu().numpy().flatten()

        if len(prediction) == prediction_size:
            stock_data.loc[stock_data.index[end_index:end_index+prediction_size], 'close'] = prediction

        # 保存实际值和预测值
        actual_values = stock_data['close'].values[end_index:end_index+prediction_size]
        if len(actual_values) != len(prediction):
            continue  # 跳过长度不匹配的情况
        all_actual_values.extend(actual_values)
        all_predictions.extend(prediction)
        all_dates.extend(stock_data['trade_date'].values[end_index:end_index+prediction_size])

        # 计算评估指标
        mape = mean_absolute_percentage_error(actual_values, prediction)
        mse = mean_squared_error(actual_values, prediction)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(actual_values, prediction)
        r2 = r2_score(actual_values, prediction)
        msle = mean_squared_log_error(actual_values, prediction)

        # 计算未来一周的收益率
        future_return = (prediction[-1] - stock_data['open'].values[end_index]) / stock_data['open'].values[end_index]
        stock_returns.append((stock_code, future_return))

        # 保存当前股票代码的预测数据和评估指标
        current_results[stock_code] = {
            'dates': stock_data['trade_date'].values[end_index:end_index+prediction_size],
            'actual_values': actual_values,
            'predictions': prediction
            }
        metrics.append([stock_code, mape, mse, rmse, mae, r2, msle])

    # 按未来收益率排序，选取收益最高的10只股票
    stock_returns.sort(key=lambda x: x[1], reverse=True)
    top_stocks = stock_returns[:10]

    # 计算投资组合价值
    for stock_code, _ in top_stocks:
        stock_data = all_datas[all_datas['ts_code'] == stock_code]
        start_index = stock_data[stock_data['trade_date'] >= current_date].index[0]
        end_index = start_index + window_size
        buy_price = stock_data['open'].values[end_index]
        sell_price = stock_data['close'].values[end_index + prediction_size - 1]
        cash -= buy_price
        cash += sell_price

    portfolio_value.append(cash)
    portfolio_dates.append(current_date)

    # 保存当前日期的结果
    results[current_date.date()] = current_results

    current_date += pd.Timedelta(days=7)  # 时间点A往后前进一周

# 保存训练结果到文件
with open('training_results.pkl', 'wb') as f:
    pickle.dump({
        'all_actual_values': all_actual_values,
        'all_predictions': all_predictions,
        'all_dates': all_dates,
        'metrics': metrics,
        'results': results,
        'portfolio_value': portfolio_value,
        'portfolio_dates': portfolio_dates
    }, f)

# 加载训练结果
with open('training_results.pkl', 'rb') as f:
    data = pickle.load(f)

# 确保 all_actual_values 和 all_predictions 的长度一致
all_actual_values = np.array(all_actual_values)
all_predictions = np.array(all_predictions)
all_dates = np.array(all_dates)

# 创建评估指标的 DataFrame
metrics_df = pd.DataFrame(metrics, columns=['Stock Code', 'MAPE', 'MSE', 'RMSE', 'MAE', 'R²', 'MSLE'])
print(metrics_df)

# 保存评估指标到 CSV 文件
metrics_df.to_csv('stock_metrics.csv', index=False)

# 为每个股票代码单独绘制图表
for stock_code, data in results.items():
    plt.figure(figsize=(15, 6), dpi=150)
    plt.plot(data['dates'], data['actual_values'], color='blue', lw=2)
    plt.plot(data['dates'], data['predictions'], color='red', lw=2)
    plt.title(f'Stock Price Prediction for {stock_code}', fontsize=15)
    plt.xlabel('Date', fontsize=12)
    plt.ylabel('Price', fontsize=12)
    plt.legend(['Actual Price', 'Predicted Price'], loc='upper left', prop={'size': 15})
    plt.grid(color='white')
    plt.savefig(f'{stock_code}_prediction.png')
    plt.close()

# 保存结果到文件
with open('stock_predictions_CNN.pkl', 'wb') as f:
    pickle.dump(results, f)