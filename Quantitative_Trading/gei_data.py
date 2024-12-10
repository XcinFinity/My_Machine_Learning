import pandas as pd
import time
import glob
from datetime import datetime

import yfinance as yf

# Tushare Pro API,  https://tushare.pro
import tushare as ts
ts.set_token('3d35a3730c43014dec0df58ec056ce7cbd9e5a54a1727e0f15649592')
pro = ts.pro_api()

today = datetime.today().strftime('%Y%m%d')
yesterday = (datetime.today() - pd.Timedelta(days=1)).strftime('%Y%m%d')

# def get_A500_code():
#     index_code = '000510.SH'
#     data = pro.index_member(index_code=index_code, start_date=yesterday, end_date=today)
#     return data

# Code_A500 = get_A500_code()
# print(Code_A500)

def get_A500_code_from_xlsx(file_path):
    df = pd.read_excel(file_path, dtype=str)  # 将所有数据读取为字符串类型
    stock_codes = df['Stock Code'].tolist()
    
    # 添加后缀
    for i in range(len(stock_codes)):
        stock_codes[i] = str(stock_codes[i])  # 确保股票代码是字符串
        if stock_codes[i].startswith('0') or stock_codes[i].startswith('3'):
            stock_codes[i] += '.SZ'
        elif stock_codes[i].startswith('6'):
            stock_codes[i] += '.SH'
    
    return stock_codes

file_path = '/Users/lyndonbin/Downloads/VS_Code/data/Quantitative-Trading/A500.xlsx'
Code_A500 = get_A500_code_from_xlsx(file_path)
print(Code_A500)

# 下載數據
def download_data():
    total_codes = len(Code_A500)

    for i in range(len(Code_A500)):
        code = Code_A500[i]
        data = pro.stk_factor_pro(ts_code=code, start_date='20000101', end_date=today)
        # print(data.head())
        
        file_name = f'/Users/lyndonbin/Downloads/VS_Code/data/Quantitative-Trading/A500/{code}_data.csv'
        data.to_csv(file_name, index=False)
        
        progress = (i + 1) / total_codes * 100
        print(f'正在处理 {code} ({i + 1}/{total_codes}) - 进度: {progress:.2f}%')
        
        time.sleep(10)

# 整合數據
def SSE_A500_data():
    sse_data = glob.glob('/Users/lyndonbin/Downloads/VS_Code/data/Quantitative-Trading/A500/*_data.csv')

    SSE_50_data = pd.DataFrame()

    for file in sse_data:
        data = pd.read_csv(file)
        SSE_50_data = pd.concat([SSE_50_data, data], axis=0)

    SSE_50_data.to_csv('/Users/lyndonbin/Downloads/VS_Code/data/Quantitative-Trading/A500_data.csv', index=False)

download_data()
SSE_A500_data()