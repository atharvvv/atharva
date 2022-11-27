#!C:\Users\Lenovo\AppData\Local\Programs\Python\Python37-32\python.exe

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
import warnings
import pickle
warnings.filterwarnings("ignore")

lifedata = pd.read_csv("Life.csv")
lifedata = np.array(lifedata)

y=lifedata['Life expectancy']
x=lifedata.drop('Life expectancy',axis=1)

Country_dummy=pd.get_dummies(x['Country'])
status_dummy=pd.get_dummies(x['Status'])
x.drop(['Country','Status'],inplace=True,axis=1)
x=pd.concat([x,Country_dummy,status_dummy],axis=1)

print(x)
print(y)

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state ="0")


print(x_train)

print(x_test)

print(y_train)

print(y_test)

x_train = x_train.dropna()
y_train = y_train.dropna()

y_train = y_train.values.reshape(-1,1)
x_train = x_train.values.reshape(-1,1)
x_test = x_test.values.reshape(-1,1)
y_test = y_test.values.reshape(-1,1)



from sklearn.ensemble import RandomForestRegressor
regressor = RandomForestRegressor(n_estimators = 100)

# fit the regressor with x and y data
regressor.fit(x_train, y_train)
regressor.score(x_train,y_train)

pickle.dump(regressor, open('moel.pk1', 'wb'))
pickle.dump(regressor, open('moel.pk1', 'rb'))