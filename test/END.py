import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pickle


# 加载训练结果
with open('training_results.pkl', 'rb') as f:
    data = pickle.load(f)

start_date = pd.to_datetime('2004-01-03')
end_date = pd.to_datetime('2024-01-01')
current_date = start_date
window_size = 252 * 3
prediction_size = 5

# 初始化交易系统
initial_cash = 10000
cash = initial_cash

all_actual_values = data['all_actual_values']
all_predictions = data['all_predictions']
all_dates = data['all_dates']
metrics = data['metrics']
results = data['results']
portfolio_value = data['portfolio_value']
portfolio_dates = data['portfolio_dates']

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

# 计算投资组合评估指标
def annualized_return(portfolio_value, start_date, end_date):
    total_return = (portfolio_value[-1] / portfolio_value[0]) - 1
    num_years = (end_date - start_date).days / 365.25
    return (1 + total_return) ** (1 / num_years) - 1

def annualized_volatility(portfolio_value):
    returns = np.diff(portfolio_value) / portfolio_value[:-1]
    return np.std(returns) * np.sqrt(252)

def sharpe_ratio(portfolio_value, risk_free_rate=0.01):
    returns = np.diff(portfolio_value) / portfolio_value[:-1]
    excess_returns = returns - risk_free_rate / 252
    return np.mean(excess_returns) / np.std(excess_returns) * np.sqrt(252)

def max_drawdown(portfolio_value):
    peak = np.maximum.accumulate(portfolio_value)
    drawdown = (portfolio_value - peak) / peak
    return np.min(drawdown)

# 计算并打印投资组合评估指标
annual_return = annualized_return(portfolio_value, start_date, end_date)
volatility = annualized_volatility(portfolio_value)
sharpe = sharpe_ratio(portfolio_value)
max_dd = max_drawdown(portfolio_value)

print(f'Annualized Return: {annual_return * 100:.2f}%')
print(f'Annualized Volatility: {volatility * 100:.2f}%')
print(f'Sharpe Ratio: {sharpe:.2f}')
print(f'Max Drawdown: {max_dd * 100:.2f}%')

# 绘制投资组合价值曲线
plt.figure(figsize=(15, 6), dpi=150)
plt.plot(portfolio_dates, portfolio_value, color='green', lw=2)
plt.title('Portfolio Value Over Time', fontsize=15)
plt.xlabel('Date', fontsize=12)
plt.ylabel('Portfolio Value', fontsize=12)
plt.grid(color='white')
plt.show()

# 打印最终投资组合价值
print(f'Initial Cash: {initial_cash}')
print(f'Final Portfolio Value: {portfolio_value[-1]}')
print(f'Return: {(portfolio_value[-1] - initial_cash) / initial_cash * 100:.2f}%')