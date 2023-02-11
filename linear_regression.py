import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

df = pd.read_csv('data.csv')
X = df.drop('Weight', axis = 1)
y = df['Weight']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)
print('Train dimension: ', X_train.shape)
print('Test dimension: ', X_test.shape)

model = LinearRegression()
model.fit(X_train, y_train)
print('Coeficient: ', model.coef_)
print('Interception: ', model.intercept_)

y_pred = model.predict(X_test)
MSE = mean_squared_error(y_test, y_pred)
print("MSE:", MSE)

plt.scatter(X, y, color = "blue")
plt.plot(X_test, y_pred, color = "red")
plt.xlabel("Height")
plt.ylabel("Weight")
plt.show()