# https://archive.ics.uci.edu/dataset/159/magic+gamma+telescope
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_openml
from ucimlrepo import fetch_ucirepo


# magic = fetch_openml(data_id=1199, as_frame=True)
# x=magic.data
# y=magic.target
#
# print(x.head()) #features
# print("target")
# print(y.head())# target

cols=['flength','fwidth','fsize','fconc','fConc1','fAsym','fM3Long','fM3Trans','fAlpha','fDist','class']
data =pd.read_csv("Data/magic+gamma+telescope/magic04.data",names=cols)

# print(data.head())
# print(data['class'].unique()) # g-gamma and h-hadron

data['class']= (data['class']== 'g').astype(int) # convert to 0 and 1
print(data['class'].unique()) # 1-gamma and 0-hadron
print(data.head())

# for label in cols[:-1]:
#     plt.figure()
#     plt.hist(data[data['class']==1][label],bins=30,alpha=0.5,label='gamma')
#     plt.hist(data[data['class']==0][label],bins=30,alpha=0.5,label='hadron')
#     plt.xlabel(label)
#     plt.ylabel('count')
#     plt.legend()
#     plt.show()

# Split Dataset into Train Vali Test

train,valid,test = np.split(data.sample(frac=1),[int(.6*len(data)),int(.8*len(data))])
print(len(train),len(valid),len(test))
