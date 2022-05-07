# 1 import library
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import clf
from sklearn.impute import SimpleImputer

#2 import csv file using pandas
df = pd.read_csv('C:\weather/rainfall in india 1901-2015.csv')
sol = df.head(100)
df_all = sol.fillna(0)
corr2 = df_all.corr()
#print(corr2)


# 3 making some change for clean data
newarr = df_all['ANNUAL']
newarr2 = df_all['YEAR']

array = np.array([newarr])
array_2 = np.array([newarr2])

# 4 Reshaping data in 2D array
x = array.reshape(-1, 1)
y = array_2.reshape(-1, 1)

# 5 using train and test method
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2)

# 6 using linear regression
clf = LinearRegression()
clf.fit(x_train, y_train)
ans = clf.predict(y_test)
print(ans[1])

