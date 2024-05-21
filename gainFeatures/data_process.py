"""
Normalisation of feature files
"""

import pandas as pd
import numpy as np
from sklearn.utils import shuffle

excel_raw_path = './DataSets/seg_model/features/xgb_excel_raw_10.xlsx'
excel_normed_path = './DataSets/seg_model/features/xgb_excel_normed_10.xlsx'


label_maps = {'0':0, '1':1, '2':2, '4':3}
data_frame = pd.read_excel(excel_raw_path)
target = 'zLabel'
predictors = [x for x in data_frame.columns if x not in [target]]

"""
Upper and lower intervals for feature scaling
"""
weights = data_frame[predictors].apply(lambda x: [np.min(x),np.max(x)])
weights.to_csv('./DataSets/seg_model/ModelWeights/xgb_excel_scale_10.csv', index=None) # Upper and lower intervals for feature scaling

df_normed = data_frame[predictors].apply(lambda x: (x - np.min(x)) / (np.max(x) - np.min(x)))
df_normed[target] = data_frame[target]
df_normed[target] = [label_maps[str(s)] for s in df_normed[target]]
df_normed = shuffle(df_normed)
df_normed.fillna(0, inplace=True)

n_writer = pd.ExcelWriter(excel_normed_path)
df_normed.to_excel(n_writer, index=False, encoding='utf-8')
n_writer.save()
print("The excel file was saved successfully!")
"""
xlsx to csv
"""
data_xls = pd.read_excel('./DataSets/seg_model/features/xgb_excel_normed_10.xlsx', index_col=0)     #输入xlsx文件名
data_xls.to_csv('../data/data.csv', encoding='utf-8')