import numpy as np
import pandas as pd
import sklearn as sk
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt

df = pd.read_pickle("allData.pkl")
df_test = df[:10]

x = df_test['Timestamp']
y= df_test['Open']
# plt.scatter(x, y)
# plt.show()
X = x[:, np.newaxis]
print(X)
model = LinearRegression(fit_intercept=True)
model.fit(X, y)
a = model.coef_
print("a = ", a[0])
b = model.intercept_
print("b = ", b)

xnew = np.array([1325600520])
ynew = model.predict(xnew.reshape(-1,1))
print(ynew)

print(df_test)