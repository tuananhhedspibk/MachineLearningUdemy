# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('Salary_Data.csv')
X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, 1].values

# Splitting the dataset into the Training set and Test set
from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state = 0)

# Fitting Simple Linear Regression to the Training set
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X_train, y_train) # learn collaboration between X, y

# Predicting and test set results
y_pred = regressor.predict(X_test)

# Visualize the Training set results
plt.scatter(X_train, y_train, color = 'red') # make point(x_train, y_train)
plt.plot(X_train, regressor.predict(X_train), color = 'blue') # draw line
plt.title('Salary vs Experience (Training Set)')
plt.xlabel('Years of experience')
plt.ylabel('Salary')
plt.show()

plt.scatter(X_test, y_test, color = 'red') # make point(x_train, y_train)
plt.plot(X_train, regressor.predict(X_train), color = 'blue') # draw line
plt.title('Salary vs Experience (Test Set)')
plt.xlabel('Years of experience')
plt.ylabel('Salary')
plt.show()