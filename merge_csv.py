import os
import glob
import pandas as pd
csv_dir = './data_gather/csv_files'
output = './data_gather/spectral_data.csv'
data = pd.DataFrame()
for lujing in glob.glob(os.path.join(csv_dir, "*.csv")):
    df = pd.read_csv(lujing, dtype='float')
    data = pd.concat([data, df], axis=0)

for i in range(data.shape[0]):
    data.iloc[i][0] = i + 1

print(data)  # 看看数据
data.to_csv(output, index=False)  # 保存吧
