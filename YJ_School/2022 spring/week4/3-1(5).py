import pandas as pd
from pandas import Series, DataFrame
data=pd.read_csv('C:/Users/User/OneDrive/문서/Pycharm/BI Lab/week4/bee_data_crop.csv')
dataf=DataFrame(data)
X=DataFrame(data, columns=['Period', 'State', 'Pesticide', 'Varroa', 'Otherpest', 'Diseases'])
Y=data['Lost Colonies']