# Python code to illustrate
# regression using data set
import matplotlib.pyplot as plt
import numpy as np
from sklearn import datasets, linear_model
import pandas as pd

# Load CSV and columns
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

df = pd.read_csv("https://raw.githubusercontent.com/Cibah/machine-learning/supervised-learning/datasets/csv-files/Housing.csv")

Y = df['price']
X = df['lotsize']

X=X.values.reshape(len(X),1)
Y=Y.values.reshape(len(Y),1)

# Split the data into training/testing sets
X_train = X[:-250]
X_test = X[-250:]

# Split the targets into training/testing sets
Y_train = Y[:-250]
Y_test = Y[-250:]

# Plot outputs
plt.scatter(X_test, Y_test, color='black')
plt.title('Test Data')
plt.xlabel('Size')
plt.ylabel('Price')
#plt.xticks(())
#plt.yticks(())


#1. Linear Regression
#2. Logistic Regression
#3. Ridge Regression
#4. Lasso Regression
#5. Polynomial Regression
#6. Bayesian Linear Regression
#7. Stepwise Regression
#8. ElasticNet Regression


# Create linear regression object
#regr = linear_model.LinearRegression()
# Create  regression object
#regr = linear_model.Lasso()
#regr = linear_model.Perceptron()
#regr = linear_model.ARDRegression()
#regr = linear_model.BayesianRidge()
regr = linear_model.ElasticNet()
#regr = linear_model.LogisticRegression()
#regr = linear_model.Ridge()




# Train the model using the training sets
regr.fit(X_train, Y_train)

# Plot outputs
plt.plot(X_test, regr.predict(X_test), color='red',linewidth=3)
plt.show()