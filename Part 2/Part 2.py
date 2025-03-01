import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

dataset = pd.read_csv("Salary_Data.csv")
X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, -1].values

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Simple Linear Regression
regressor = LinearRegression()

# Train it on the training set
regressor.fit(X_train, y_train)

y_predicted = regressor.predict(X_test)

# plt.scatter(X_train, y_train, color="red")
# plt.plot(X_train, regressor.predict(X_train), color="blue")
# plt.title("Salary vs Experience")
# plt.xlabel("Years")
# plt.ylabel("Salary")
# plt.show()

plt.scatter(X_test, y_test, color="red")
plt.plot(X_train, regressor.predict(X_train), color="blue")
plt.title("Salary vs Experience")
plt.xlabel("Years")
plt.ylabel("Salary")
plt.show()

print(regressor.predict([[12]]))
print(regressor.coef_)
print(regressor.intercept_)

