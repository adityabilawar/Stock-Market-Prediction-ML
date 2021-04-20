import pandas as pd
import numpy as np
import sklearn
from sklearn import linear_model
from sklearn.utils import shuffle
import matplotlib.pyplot as plt
import pickle
from matplotlib import style
   


data = pd.read_csv("EOD-MSFT.csv", sep=",")

data = data[["Date", "Open", "Close"]]

predict = "Close"




X = np.array(data.drop([predict], 1))
y = np.array(data[predict])

best = 0
for _ in range(30):
    x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(X, y, test_size = 0.1)

linear = linear_model.LinearRegression()

linear.fit(x_train, y_train)
acc = linear.score(x_test, y_test)
print(acc)


if acc > best:
    best = acc
with open("StockPricePredict.pickle", "wb") as f:
    pickle.dump(linear, f) 

pickle_in = open("StockPricePredict.pickle", "rb")
linear = pickle.load(pickle_in)

print('Coefficient: \n', linear.coef_)
print('Intercept: \n', linear.intercept_)

predictions = linear.predict(x_test)

for x in range(len(predictions)):
    print(predictions[x], x_test[x], y_test[x])

p = "Date"
#p = "G2" 
#p = "studytime"
#p = "failures"
#p = "absences"
style.use("ggplot")
#plt.scatter(data[p], data["Close"])
plt.title('Stock Price Prediction Model')
plt.xlabel(p)
plt.ylabel("Close Price USD ($)")
plt.plot(data["Close"])
plt.legend(["Close"], loc='lower right')
plt.show()