import pandas as pd
import numpy as np
import pickle
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error

# Fetch Salary.csv
data = pd.read_csv('Salary.csv')

# Check if value is null
print(data.isnull().sum())

print(data.info)

print(data.describe())

plt.scatter(data['YearsExperience'],data['Salary'] )
plt.xlabel('Year of Exp')
plt.ylabel('Salary')
plt.show()

X = data.drop('Salary', axis=1)
y = data['Salary']

X_train, X_test, Y_train, Y_test = train_test_split(X, y, random_state=101, test_size=0.2)
print(X_train.shape, X_test.shape, Y_train.shape, Y_test.shape)

lr = LinearRegression()
lr.fit(X_train, Y_train)


with open('trained_model.pkl', 'wb') as pickle_file:
    pickle.dump(lr, pickle_file)
