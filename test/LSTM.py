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


# 读取指数成分变更数据
IDX_Chgsmp_50 = pd.read_csv('IDX_Chgsmp.csv')

IDX_Chgsmp_50['Chgsmp01'] = pd.to_datetime(IDX_Chgsmp_50['Chgsmp01'])
IDX_Chgsmp_50['Chgsmp02'] = IDX_Chgsmp_50['Chgsmp02'].apply(lambda x: str(x) + '.SH')
IDX_Chgsmp_50['Chgsmp04'] = IDX_Chgsmp_50['Chgsmp04'].astype(int)
print(IDX_Chgsmp_50)

# 读取股票数据
all_data = pd.read_csv('SSE_50_DATA.csv')

all_datas = all_data.groupby('ts_code').apply(lambda x: x.sort_values('trade_date')).reset_index(drop = True)
all_datas['trade_date'] = pd.to_datetime(all_datas['trade_date'], format = '%Y%m%d')

# all_datas.fillna(np.nan, inplace = True)
all_datas.fillna(0, inplace=True)
# all_datas.interpolate(method='linear', inplace=True)
print(all_datas)

all_datas.columns

all_datas.duplicated().sum()

all_datas.isnull().sum().sum()

# 初始化股票代码列表
stock_change = set(all_datas['ts_code'].unique())

# stock_feature = all_datas[['close', 
#                            'open', 'high', 'low', 'pre_close', 'change', 'pct_chg', 'vol', 'amount'
#                            ]]

stock_feature = all_datas[['close', 
                            'open', 'high', 'low', 'pre_close', 'change', 'pct_chg', 'vol', 'amount', 
                            'turnover_rate', 'turnover_rate_f', 'volume_ratio', 'pe', 'pe_ttm', 'pb', 'ps', 'ps_ttm', 'dv_ratio', 'dv_ttm', 'total_share', 'float_share', 'free_share', 'total_mv', 'circ_mv', 'adj_factor', 
                            'asi_bfq', 'asit_bfq', 'atr_bfq', 'bbi_bfq', 'bias1_bfq', 'bias2_bfq', 'bias3_bfq', 'boll_lower_bfq', 'boll_mid_bfq', 'boll_upper_bfq', 
                            'brar_ar_bfq', 'brar_br_bfq', 'cci_bfq', 'cr_bfq', 'dfma_dif_bfq', 'dfma_difma_bfq', 'dmi_adx_bfq', 'dmi_adxr_bfq', 'dmi_mdi_bfq', 'dmi_pdi_bfq', 
                            'downdays', 'updays', 'dpo_bfq', 'madpo_bfq', 'ema_bfq_10', 'ema_bfq_20', 'ema_bfq_250', 'ema_bfq_30', 'ema_bfq_5', 'ema_bfq_60', 'ema_bfq_90', 
                            'emv_bfq', 'maemv_bfq', 'expma_12_bfq', 'expma_50_bfq', 'kdj_bfq', 'kdj_d_bfq', 'kdj_k_bfq', 'ktn_down_bfq', 'ktn_mid_bfq', 'ktn_upper_bfq', 
                            'lowdays', 'topdays', 'ma_bfq_10', 'ma_bfq_20', 'ma_bfq_250', 'ma_bfq_30', 'ma_bfq_5', 'ma_bfq_60', 'ma_bfq_90', 'macd_bfq', 'macd_dea_bfq', 'macd_dif_bfq', 
                            'mass_bfq', 'ma_mass_bfq', 'mfi_bfq', 'mtm_bfq', 'mtmma_bfq', 'obv_bfq', 'psy_bfq', 'psyma_bfq', 'roc_bfq', 'maroc_bfq', 'rsi_bfq_12', 'rsi_bfq_24', 'rsi_bfq_6', 
                            'taq_down_bfq', 'taq_mid_bfq', 'taq_up_bfq', 'trix_bfq', 'trma_bfq', 'vr_bfq', 'wr_bfq', 'wr1_bfq', 'xsii_td1_bfq', 'xsii_td2_bfq', 'xsii_td3_bfq', 'xsii_td4_bfq'
                            ]]


# 定义 LSTM 模型
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

        X_train, y_train = prepare_data(start_index, end_index, window_size, prediction_size, scaled_features, stock_data)
        if len(X_train) == 0 or len(y_train) == 0:
            continue  # 跳过没有有效训练数据的情况
        train_dataset = TensorDataset(torch.tensor(X_train, dtype=torch.float32), torch.tensor(y_train, dtype=torch.float32))
        train_loader = DataLoader(train_dataset, batch_size=100, shuffle=True)
        
        optimizer = optim.Adam(model.parameters(), lr=0.001)

        train_model(model, train_loader, criterion, optimizer, device, current_date, stock_code)
        prediction = predict(model, scaled_features, end_index, window_size, input_size, device)

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
with open('training_results_LSTM.pkl', 'wb') as f:
    pickle.dump({
        'all_actual_values': all_actual_values,
        'all_predictions': all_predictions,
        'all_dates': all_dates,
        'metrics': metrics,
        'results': results,
        'portfolio_value': portfolio_value,
        'portfolio_dates': portfolio_dates
    }, f)


# # 加载训练结果
# with open('training_results.pkl', 'rb') as f:
#     data = pickle.load(f)

# # 确保 all_actual_values 和 all_predictions 的长度一致
# all_actual_values = np.array(all_actual_values)
# all_predictions = np.array(all_predictions)
# all_dates = np.array(all_dates)

# # 创建评估指标的 DataFrame
# metrics_df = pd.DataFrame(metrics, columns=['Stock Code', 'MAPE', 'MSE', 'RMSE', 'MAE', 'R²', 'MSLE'])
# print(metrics_df)

# # 保存评估指标到 CSV 文件
# metrics_df.to_csv('stock_metrics.csv', index=False)

# # 为每个股票代码单独绘制图表
# for stock_code, data in results.items():
#     plt.figure(figsize=(15, 6), dpi=150)
#     plt.plot(data['dates'], data['actual_values'], color='blue', lw=2)
#     plt.plot(data['dates'], data['predictions'], color='red', lw=2)
#     plt.title(f'Stock Price Prediction for {stock_code}', fontsize=15)
#     plt.xlabel('Date', fontsize=12)
#     plt.ylabel('Price', fontsize=12)
#     plt.legend(['Actual Price', 'Predicted Price'], loc='upper left', prop={'size': 15})
#     plt.grid(color='white')
#     plt.savefig(f'{stock_code}_prediction.png')
#     plt.close()

# # 保存结果到文件
# with open('stock_predictions.pkl', 'wb') as f:
#     pickle.dump(results, f)

# # 计算投资组合评估指标
# def annualized_return(portfolio_value, start_date, end_date):
#     total_return = (portfolio_value[-1] / portfolio_value[0]) - 1
#     num_years = (end_date - start_date).days / 365.25
#     return (1 + total_return) ** (1 / num_years) - 1

# def annualized_volatility(portfolio_value):
#     returns = np.diff(portfolio_value) / portfolio_value[:-1]
#     return np.std(returns) * np.sqrt(252)

# def sharpe_ratio(portfolio_value, risk_free_rate=0.01):
#     returns = np.diff(portfolio_value) / portfolio_value[:-1]
#     excess_returns = returns - risk_free_rate / 252
#     return np.mean(excess_returns) / np.std(excess_returns) * np.sqrt(252)

# def max_drawdown(portfolio_value):
#     peak = np.maximum.accumulate(portfolio_value)
#     drawdown = (portfolio_value - peak) / peak
#     return np.min(drawdown)

# # 计算并打印投资组合评估指标
# annual_return = annualized_return(portfolio_value, start_date, end_date)
# volatility = annualized_volatility(portfolio_value)
# sharpe = sharpe_ratio(portfolio_value)
# max_dd = max_drawdown(portfolio_value)

# print(f'Annualized Return: {annual_return * 100:.2f}%')
# print(f'Annualized Volatility: {volatility * 100:.2f}%')
# print(f'Sharpe Ratio: {sharpe:.2f}')
# print(f'Max Drawdown: {max_dd * 100:.2f}%')

# # 绘制投资组合价值曲线
# plt.figure(figsize=(15, 6), dpi=150)
# plt.plot(portfolio_dates, portfolio_value, color='green', lw=2)
# plt.title('Portfolio Value Over Time', fontsize=15)
# plt.xlabel('Date', fontsize=12)
# plt.ylabel('Portfolio Value', fontsize=12)
# plt.grid(color='white')
# plt.show()

# # 打印最终投资组合价值
# print(f'Initial Cash: {initial_cash}')
# print(f'Final Portfolio Value: {portfolio_value[-1]}')
# print(f'Return: {(portfolio_value[-1] - initial_cash) / initial_cash * 100:.2f}%')