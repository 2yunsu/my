import pandas as pd
from pandas import DataFrame
data=pd.read_csv('C:/Users/User/OneDrive/문서/Pycharm/BI Lab/week4/bee_data.csv')
dataf=DataFrame(data)
sorted_data=dataf.sort_values(by='Lost Colonies',ascending=False)
not_top10=sorted_data.iloc[10: ]
not_tb10=not_top10.iloc[:-10]
not_tb10.to_csv('C:/Users/User/OneDrive/문서/Pycharm/BI Lab/week4/bee_data_crop.csv')