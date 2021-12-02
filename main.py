import numpy as np
import pandas as pd
import sklearn as sk
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt

df = pd.read_pickle("allData.pkl")
df_test = df[:100]

x = df_test['Timestamp']
y= df_test['Open']

X = x[:, np.newaxis]
model = LinearRegression(fit_intercept=True)
model.fit(X, y)

a = model.coef_
print("a = ", a[0])
b = model.intercept_
print("b = ", b)

xnew = np.array(x)
ynew = model.predict(xnew.reshape(-1,1))

plt.scatter(x, y, color='k')
plt.scatter(xnew, np.zeros(xnew.shape[0]), color='b')
plt.scatter(xnew, ynew, color='r')
plt.plot(xnew, ynew, color='m')

ypred = model.predict(X)
print(ypred.shape)
print('Erreur quadratique moyenne = ', np.mean((y-ypred)**2))

plt.show()