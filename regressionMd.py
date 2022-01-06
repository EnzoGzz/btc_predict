import numpy as np
import pandas as pd
import sklearn as sk
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
# import pickle5 as pickle


# reading

# with open("allData.pkl", "rb") as fh:  # for python < 3.8
#   bitcoin = pickle.load(fh)

bitcoin = pd.read_pickle("allData.pkl")

required_features = ['Open', 'High', 'Low', 'Volume_(BTC)', 'Volume_(Currency)', 'Weighted_Price']
output_label = 'Close'

# preprocessing the data
x_train, x_test, y_train, y_test = train_test_split(
    bitcoin[required_features],
    bitcoin[output_label],
    test_size=0.3
)

# creating the model
model = LinearRegression()

model.fit(x_train,y_train)

print(model.score(x_test,y_test))

# predicting

future_set = bitcoin.tail(30)

prediction = model.predict(future_set[required_features])

plt.figure(figsize = (12, 7))
plt.plot(bitcoin["Timestamp"][-100:-1], bitcoin["Close"][-100:-1], color='goldenrod', lw=2)
plt.plot(future_set["Timestamp"], prediction, color='deeppink', lw=2)
plt.title("Bitcoin Price over time", size=25)
plt.xlabel("Time", size=20)
plt.ylabel("$ Price", size=20)


plt.show()
