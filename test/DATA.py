import pandas as pd
import time
import glob

# Tushare Pro API,  https://tushare.pro
import tushare as ts
ts.set_token('3d35a3730c43014dec0df58ec056ce7cbd9e5a54a1727e0f15649592')
pro = ts.pro_api()

# 確定上證50成分股
Chgsmp = pd.read_csv("/Users/lyndonbin/Downloads/VS_Code/data/Machine-Learning/IDX_Chgsmp.csv")
Chgsmp['Chgsmp02'] = Chgsmp['Chgsmp02'].apply(lambda x: str(x) + '.SH')
Code_50 = Chgsmp['Chgsmp02'].drop_duplicates().tolist()

# 下載數據
def download_data():
    total_codes = len(Code_50)

    for i in range(len(Code_50)):
        code = Code_50[i]
        data = pro.stk_factor_pro(ts_code=code, start_date='20000101', end_date='20240930')
        # print(data.head())
        
        file_name = f'/Users/lyndonbin/Downloads/VS_Code/data/Machine-Learning/{code}_data.csv'
        data.to_csv(file_name, index=False)
        
        progress = (i + 1) / total_codes * 100
        print(f'正在处理 {code} ({i + 1}/{total_codes}) - 进度: {progress:.2f}%')
        
        time.sleep(60)

# 整合數據
def SSE_50_data():
    sse_data = glob.glob('/Users/lyndonbin/Downloads/VS_Code/data/Machine-Learning/*.SH_data.csv')

    SSE_50_data = pd.DataFrame()

    for file in sse_data:
        data = pd.read_csv(file)
        SSE_50_data = pd.concat([SSE_50_data, data], axis=0)

    SSE_50_data.to_csv('/Users/lyndonbin/Downloads/VS_Code/data/Machine-Learning/SSE_50_data.csv', index=False)

download_data()
SSE_50_data()


