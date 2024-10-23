# %%

import yfinance as yf

import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import mean_squared_error, mean_absolute_error

import warnings
warnings.filterwarnings('ignore')

# %%

# 从Yahoo Finance下载数据
# aapl_data = yf.download('AAPL', start='2010-01-01', end='2024-01-01')
# aapl_data.to_csv('/Users/lyndonbin/Downloads/VS_Code/Machine_Learning/Replication/Pre_Processed_AAPL.csv')
# print(aapl_data.head())

# 读取数据
AAPL = pd.read_csv("/Users/lyndonbin/Downloads/VS_Code/Machine_Learning/Replication/Pre_Processed_AAPL.csv")

# 定义数据集
def Dataset(Data, Date):
    Train_Data = Data['Adj Close'][Data['Date'] < Date].to_numpy()  # 训练集
    Data_Train = []  # 训练集
    Data_Train_X = []  # 训练集X
    Data_Train_Y = []  # 训练集Y
    
    # 设定时间窗口为5
    for i in range(0, len(Train_Data), 5):  # 每5个数据为一个窗口
        try:
            Data_Train.append(Train_Data[i : i + 5])  # 将数据添加到训练集
        except:
            pass

    # 如果最后一个窗口不足5个数据，则删除
    if len(Data_Train[-1]) < 5:
        Data_Train.pop(-1)
    
    # 将数据集分为X和Y
    Data_Train_X = Data_Train[0 : -1]  # 训练集X
    Data_Train_X = np.array(Data_Train_X)  # 转换为数组
    Data_Train_X = Data_Train_X.reshape((-1, 5, 1))  # 转换为张量
    Data_Train_Y = Data_Train[1 : len(Data_Train)]
    Data_Train_Y = np.array(Data_Train_Y)
    Data_Train_Y = Data_Train_Y.reshape((-1, 5, 1))

    # 设定测试集
    Test_Data = Data['Adj Close'][Data['Date'] >= Date].to_numpy()
    Data_Test = []  
    Data_Test_X = []
    Data_Test_Y = []
    for i in range(0, len(Test_Data), 5):
        try:
            Data_Test.append(Test_Data[i : i + 5])
        except:
            pass

    if len(Data_Test[-1]) < 5:
        Data_Test.pop(-1)
    
    Data_Test_X = Data_Test[0 : -1]
    Data_Test_X = np.array(Data_Test_X)
    Data_Test_X = Data_Test_X.reshape((-1, 5, 1))
    Data_Test_Y = Data_Test[1 : len(Data_Test)]
    Data_Test_Y = np.array(Data_Test_Y)
    Data_Test_Y = Data_Test_Y.reshape((-1, 5, 1))

    return Data_Train_X, Data_Train_Y, Data_Test_X, Data_Test_Y  # 返回训练集和测试集

AAPL.head()
AAPL.info()

# 创建LSTM模型
class LSTMModel(nn.Module):
    def __init__(self):  # 初始化
        super(LSTMModel, self).__init__()  # 继承父类
        self.lstm1 = nn.LSTM(1, 200, batch_first=True)  # LSTM层
        self.lstm2 = nn.LSTM(200, 200, batch_first=True)
        self.fc1 = nn.Linear(200, 200)  # 全连接层
        self.fc2 = nn.Linear(200, 100)
        self.fc3 = nn.Linear(100, 50)
        self.fc4 = nn.Linear(50, 5)
        self.leaky_relu = nn.LeakyReLU()  # 激活函数

    # 前向传播
    def forward(self, x):
        x, _ = self.lstm1(x)  # LSTM层
        x, _ = self.lstm2(x)
        x = self.leaky_relu(self.fc1(x[:, -1, :]))  # 全连接层
        x = self.leaky_relu(self.fc2(x))
        x = self.leaky_relu(self.fc3(x))
        x = self.fc4(x)
        return x  # 返回输出

# 实例化模型
model = LSTMModel()

print(model)

# 定义学习率调度器
def scheduler(epoch):  # 定义学习率调度器
    if epoch <= 150:  # 前150个epoch
        lrate = (10 ** -5) * (epoch / 150)  # 学习率为(10 ** -5) * (epoch / 150)
    elif epoch <= 400:
        initial_lrate = (10 ** -5)
        k = 0.01
        lrate = initial_lrate * math.exp(-k * (epoch - 150))
    else:
        lrate = (10 ** -6)
    return lrate

# 绘制学习率调度器
epochs = [i for i in range(1, 1001, 1)]  # 1000个epoch
lrate = [scheduler(i) for i in range(1, 1001, 1)]  # 学习率
plt.plot(epochs, lrate)  # 绘制学习率调度器

# 数据预处理
AAPL.head()
AAPL.info()

# 将日期转换为时间戳
AAPL["Date"] = pd.to_datetime(AAPL["Date"])

# 将数据集分为训练集和测试集
AAPL_Date = '2014-12-31'  # 设定分割日期
AAPL_Train_X, AAPL_Train_Y, AAPL_Test_X, AAPL_Test_Y = Dataset(AAPL, AAPL_Date)  # 分割数据集

# 将数据集转换为张量
AAPL_Train_X = torch.tensor(AAPL_Train_X, dtype=torch.float32)
AAPL_Train_Y = torch.tensor(AAPL_Train_Y, dtype=torch.float32)
AAPL_Test_X = torch.tensor(AAPL_Test_X, dtype=torch.float32)
AAPL_Test_Y = torch.tensor(AAPL_Test_Y, dtype=torch.float32)

# 训练模型
criterion = nn.MSELoss()  # 损失函数
optimizer = optim.Adam(model.parameters(), lr=1e-5)  # 优化器

scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=scheduler)  # 学习率调度器

num_epochs = 1000  # 训练1000个epoch
train_losses = []  # 训练损失
val_losses = []  # 验证损失

for epoch in range(num_epochs):
    model.train()  # 设置模型为训练模式
    optimizer.zero_grad()  # 梯度清零
    outputs = model(AAPL_Train_X)  # 前向传播
    loss = criterion(outputs, AAPL_Train_Y[:, -1, :])  # 计算损失
    loss.backward()  # 反向传播
    optimizer.step()  # 更新参数
    scheduler.step()  # 更新学习率
    
    train_losses.append(loss.item())  # 记录训练损失
    
    # 验证模型
    model.eval()  # 设置模型为验证模式
    with torch.no_grad():  # 不计算梯度
        val_outputs = model(AAPL_Test_X)  # 前向传播
        val_loss = criterion(val_outputs, AAPL_Test_Y[:, -1, :])  # 计算损失
        val_losses.append(val_loss.item())  # 记录验证损失
    
    if (epoch + 1) % 100 == 0:  # 每100个epoch打印一次损失
        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}, Val Loss: {val_loss.item():.4f}')  # 打印损失

# 绘制训练损失和验证损失
epochs = range(1, num_epochs + 1)  # epoch范围

fig, (ax1, ax2) = plt.subplots(1, 2)  # 绘制训练损失和验证损失

fig.set_figheight(5)  # 设置图形高度
fig.set_figwidth(15)  # 设置图形宽度


ax1.plot(epochs, train_losses, label='Training Loss')  # 绘制训练损失
ax1.plot(epochs, val_losses, label='Validation Loss')  # 绘制验证损失
ax1.set(xlabel="Epochs", ylabel="Loss")  # 设置x轴和y轴标签
ax1.legend()  # 添加图例

plt.show()  # 显示图形

# 预测
model.eval()  # 设置模型为验证模式
with torch.no_grad():  # 不计算梯度
    AAPL_prediction = model(AAPL_Test_X).numpy()  # 预测

# 绘制预测结果
plt.figure(figsize=(20, 5))  # 设置图形大小
plt.plot(AAPL['Date'][AAPL['Date'] < '2015-01-12'], AAPL['Adj Close'][AAPL['Date'] < '2015-01-12'], label='Training')  # 绘制训练集
plt.plot(AAPL['Date'][AAPL['Date'] >= '2015-01-09'], AAPL['Adj Close'][AAPL['Date'] >= '2015-01-09'], label='Testing')  # 绘制测试集
plt.plot(AAPL['Date'][AAPL['Date'] >= '2015-01-12'], AAPL_prediction.reshape(-1), label='Predictions')  #绘制预测
plt.xlabel('Time')  # x轴标签
plt.ylabel('Closing Price')  # y轴标签
plt.legend(loc='best')  # 添加图例

# 计算RMSE和MAPE
# rmse = math.sqrt(mean_squared_error(AAPL_Test_Y.reshape(-1, 5), AAPL_prediction))  # 计算RMSE
# mape = np.mean(np.abs(AAPL_prediction - AAPL_Test_Y.reshape(-1, 5)) / np.abs(AAPL_Test_Y.reshape(-1, 5)))  # 计算MAPE
# print(f'RMSE: {rmse}')
# print(f'MAPE: {mape}')
