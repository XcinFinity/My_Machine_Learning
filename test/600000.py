import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_percentage_error


import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset


# 读取指数成分变更数据
IDX_Chgsmp_50 = pd.read_csv('/Users/lyndonbin/Downloads/VS_Code/data/Machine-Learning/IDX_Chgsmp.csv')

IDX_Chgsmp_50['Chgsmp01'] = pd.to_datetime(IDX_Chgsmp_50['Chgsmp01'])
IDX_Chgsmp_50['Chgsmp02'] = IDX_Chgsmp_50['Chgsmp02'].apply(lambda x: str(x) + '.SH')
IDX_Chgsmp_50['Chgsmp04'] = IDX_Chgsmp_50['Chgsmp04'].astype(int)
# print(IDX_Chgsmp_50.head())


# 读取股票数据
all_data = pd.read_csv('/Users/lyndonbin/Downloads/VS_Code/data/Machine-Learning/600000.SH_data.csv')

# all_data.fillna(np.nan, inplace = True)
all_data.fillna(0, inplace=True)
# all_data.interpolate(method='linear', inplace=True)

all_datas = all_data.groupby('ts_code').apply(lambda x: x.sort_values('trade_date')).reset_index(drop = True)
all_datas['trade_date'] = pd.to_datetime(all_datas['trade_date'], format = '%Y%m%d')
all_datas.interpolate(method = 'linear', inplace = True)
# print(all_datas.head())

all_datas.columns

all_datas.duplicated().sum()

all_datas.isnull().sum().sum()


features = all_datas[['close', 
                      'open', 'high', 'low', 'pre_close', 'change', 'pct_chg', 'vol', 'amount']]
# 'close', 
# 'open', 'open_hfq', 'high', 'high_hfq', 'low', 'low_hfq', 'close_hfq', 'pre_close', 'change', 'pct_chg', 'vol', 'amount', 
# 'turnover_rate', 'turnover_rate_f', 'volume_ratio', 'pe', 'pe_ttm', 'pb', 'ps', 'ps_ttm', 'dv_ratio', 'dv_ttm', 'total_share', 'float_share', 'free_share', 'total_mv', 'circ_mv', 'adj_factor', 'asi_bfq', 'asi_hfq', 'asit_bfq', 'asit_hfq', 
# 'atr_bfq', 'atr_hfq', 'bbi_bfq', 'bbi_hfq', 'bias1_bfq', 'bias1_hfq', 'bias2_bfq', 'bias2_hfq', 'bias3_bfq', 'bias3_hfq', 'boll_lower_bfq', 'boll_lower_hfq', 'boll_mid_bfq', 'boll_mid_hfq', 'boll_upper_bfq', 'boll_upper_hfq', 
# 'brar_ar_bfq', 'brar_ar_hfq', 'brar_br_bfq', 'brar_br_hfq', 'cci_bfq', 'cci_hfq', 'cr_bfq', 'cr_hfq', 'dfma_dif_bfq', 'dfma_dif_hfq', 'dfma_difma_bfq', 'dfma_difma_hfq', 'dmi_adx_bfq', 'dmi_adx_hfq', 'dmi_adxr_bfq', 'dmi_adxr_hfq', 'dmi_mdi_bfq', 'dmi_mdi_hfq', 'dmi_pdi_bfq', 
# 'dmi_pdi_hfq', 'downdays', 'updays', 'dpo_bfq', 'dpo_hfq', 'madpo_bfq', 'madpo_hfq', 'ema_bfq_10', 'ema_bfq_20', 'ema_bfq_250', 'ema_bfq_30', 'ema_bfq_5', 'ema_bfq_60', 'ema_bfq_90', 'ema_hfq_10', 'ema_hfq_20', 'ema_hfq_250', 'ema_hfq_30', 'ema_hfq_5', 'ema_hfq_60', 'ema_hfq_90', 
# 'emv_bfq', 'emv_hfq', 'maemv_bfq', 'maemv_hfq', 'expma_12_bfq', 'expma_12_hfq', 'expma_50_bfq',  'expma_50_hfq', 'kdj_bfq', 'kdj_hfq', 'kdj_d_bfq', 'kdj_d_hfq', 'kdj_k_bfq', 'kdj_k_hfq', 'ktn_down_bfq', 'ktn_down_hfq', 'ktn_mid_bfq', 'ktn_mid_hfq', 'ktn_upper_bfq', 'ktn_upper_hfq', 
# 'lowdays', 'topdays', 'ma_bfq_10', 'ma_bfq_20', 'ma_bfq_250', 'ma_bfq_30', 'ma_bfq_5', 'ma_bfq_60', 'ma_bfq_90', 'ma_hfq_10', 'ma_hfq_20', 'ma_hfq_250', 'ma_hfq_30', 'ma_hfq_5', 'ma_hfq_60', 'ma_hfq_90', 'macd_bfq', 'macd_hfq', 'macd_dea_bfq', 'macd_dea_hfq', 'macd_dif_bfq', 'macd_dif_hfq', 
#            'mass_bfq', 'mass_hfq', 'ma_mass_bfq', 'ma_mass_hfq', 'mfi_bfq', 'mfi_hfq', 'mtm_bfq', 'mtm_hfq', 'mtmma_bfq', 'mtmma_hfq', 'obv_bfq', 'obv_hfq', 'psy_bfq', 'psy_hfq', 'psyma_bfq', 'psyma_hfq', 'roc_bfq', 'roc_hfq', 'maroc_bfq', 'maroc_hfq', 'rsi_bfq_12', 'rsi_bfq_24', 'rsi_bfq_6', 'rsi_hfq_12', 'rsi_hfq_24', 'rsi_hfq_6', 
#            'taq_down_bfq', 'taq_down_hfq', 'taq_mid_bfq', 'taq_mid_hfq', 'taq_up_bfq', 'taq_up_hfq', 'trix_bfq', 'trix_hfq', 'trma_bfq', 'trma_hfq', 'vr_bfq', 'vr_hfq', 'wr_bfq', 'wr_hfq', 'wr1_bfq', 'wr1_hfq', 'xsii_td1_bfq', 'xsii_td1_hfq', 'xsii_td2_bfq', 'xsii_td2_hfq', 'xsii_td3_bfq', 'xsii_td3_hfq', 'xsii_td4_bfq', 'xsii_td4_hfq'

# all_datas[['close', 
#            'open', 'high', 'low', 'pre_close', 'change', 'pct_chg', 'vol', 'amount', 
#            'turnover_rate', 'turnover_rate_f', 'volume_ratio', 'pe', 'pe_ttm', 'pb', 'ps', 'ps_ttm', 'dv_ratio', 'dv_ttm', 'total_share', 'float_share', 'free_share', 'total_mv', 'circ_mv', 'adj_factor', 
#            'asi_bfq', 'asit_bfq', 'atr_bfq', 'bbi_bfq', 'bias1_bfq', 'bias2_bfq', 'bias3_bfq', 'boll_lower_bfq', 'boll_mid_bfq', 'boll_upper_bfq', 
#            'brar_ar_bfq', 'brar_br_bfq', 'cci_bfq', 'cr_bfq', 'dfma_dif_bfq', 'dfma_difma_bfq', 'dmi_adx_bfq', 'dmi_adxr_bfq', 'dmi_mdi_bfq', 'dmi_pdi_bfq', 
#            'downdays', 'updays', 'dpo_bfq', 'madpo_bfq', 'ema_bfq_10', 'ema_bfq_20', 'ema_bfq_250', 'ema_bfq_30', 'ema_bfq_5', 'ema_bfq_60', 'ema_bfq_90', 
#            'emv_bfq', 'maemv_bfq', 'expma_12_bfq', 'expma_50_bfq', 'kdj_bfq', 'kdj_d_bfq', 'kdj_k_bfq', 'ktn_down_bfq', 'ktn_mid_bfq', 'ktn_upper_bfq', 
#            'lowdays', 'topdays', 'ma_bfq_10', 'ma_bfq_20', 'ma_bfq_250', 'ma_bfq_30', 'ma_bfq_5', 'ma_bfq_60', 'ma_bfq_90', 'macd_bfq', 'macd_dea_bfq', 'macd_dif_bfq', 
#            'mass_bfq', 'ma_mass_bfq', 'mfi_bfq', 'mtm_bfq', 'mtmma_bfq', 'obv_bfq', 'psy_bfq', 'psyma_bfq', 'roc_bfq', 'maroc_bfq', 'rsi_bfq_12', 'rsi_bfq_24', 'rsi_bfq_6', 
#            'taq_down_bfq', 'taq_mid_bfq', 'taq_up_bfq', 'trix_bfq', 'trma_bfq', 'vr_bfq', 'wr_bfq', 'wr1_bfq', 'xsii_td1_bfq', 'xsii_td2_bfq', 'xsii_td3_bfq', 'xsii_td4_bfq'

scaler = MinMaxScaler()
scaled_features = scaler.fit_transform(features)


class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size):
        super(LSTMModel, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
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
        if i < window_size:
            continue
        X_train.append(scaled_features[i-window_size:i])
        y_train.append(all_datas['close'].values[i:i+prediction_size])  # 预测 close 价格
    return np.array(X_train), np.array(y_train)


def train_model(model, train_loader, criterion, optimizer, device, current_date):
    model.train()
    for epoch in range(100):  # 每次训练100个epoch
        running_loss = 0.0
        for batch_idx, (X_batch, y_batch) in enumerate(train_loader):
            X_batch = X_batch.to(device)
            y_batch = y_batch.to(device)
            outputs = model(X_batch)
            loss = criterion(outputs, y_batch)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        print(f'[{current_date.date()}] Epoch [{epoch+1}/100], Average Loss: {running_loss / len(train_loader):.4f}')


def predict(model, scaled_features, end_index, window_size, input_size, device):
    model.eval()
    X_test = scaled_features[end_index-window_size:end_index].reshape(1, window_size, input_size)
    X_test = torch.tensor(X_test, dtype=torch.float32).to(device)
    with torch.no_grad():
        prediction = model(X_test).cpu().numpy().flatten()
    return prediction


start_date = pd.to_datetime('2004-01-01')
end_date = pd.to_datetime('2024-01-01')
current_date = start_date
window_size = 252 * 3
prediction_size = 5
predictions = []

input_size = scaled_features.shape[1]
hidden_size = 100
num_layers = 2
output_size = prediction_size
model = LSTMModel(input_size, hidden_size, num_layers, output_size)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)


while current_date < end_date:
    start_index = all_datas[all_datas['trade_date'] >= current_date].index[0]
    end_index = start_index + window_size
    if end_index + prediction_size > len(scaled_features):
        break

    X_train, y_train = prepare_data(start_index, end_index, window_size, prediction_size, scaled_features, all_datas)
    train_dataset = TensorDataset(torch.tensor(X_train, dtype=torch.float32), torch.tensor(y_train, dtype=torch.float32))
    train_loader = DataLoader(train_dataset, batch_size=100, shuffle=True)
    train_model(model, train_loader, criterion, optimizer, device, current_date)
    prediction = predict(model, scaled_features, end_index, window_size, input_size, device)
    predictions.extend(prediction)
    all_datas.loc[end_index:end_index+prediction_size-1, 'close'] = prediction
    current_date += pd.Timedelta(days=7)  # 时间点A往后前进一周


actual_values = all_datas['close'].values[start_index + window_size:start_index + window_size + len(predictions)]
predictions = predictions[:len(actual_values)]
mape = mean_absolute_percentage_error(actual_values, predictions)
print(f'MAPE: {mape}')


plt.figure(figsize=(15, 6), dpi=150)
plt.plot(all_datas['trade_date'][start_index + window_size:start_index + window_size + len(predictions)], actual_values, color='blue', lw=2)
plt.plot(all_datas['trade_date'][start_index + window_size:start_index + window_size + len(predictions)], predictions, color='red', lw=2)
plt.title('Gold Price Prediction', fontsize=15)
plt.xlabel('Date', fontsize=12)
plt.ylabel('Price', fontsize=12)
plt.legend(['Actual Price', 'Predicted Price'], loc='upper left', prop={'size': 15})
plt.grid(color='white')
plt.show()