"""Regression Works"""

from math import sqrt
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.svm import SVR
from sklearn.metrics import r2_score, mean_squared_log_error, mean_squared_error

R_TYPE = "random_forest"

# Importing the dataset
dataset = pd.read_csv('../data/hour.csv')

# Calling Features as list
season = dataset.iloc[:, 2:3].values
year = dataset.iloc[:, 3:4].values
month = dataset.iloc[:, 4:5].values
hour = dataset.iloc[:, 5:6].values
holiday = dataset.iloc[:, 6:7].values
weekday = dataset.iloc[:, 7:8].values
workingday = dataset.iloc[:, 8:9].values
weather = dataset.iloc[:, 9:10].values
A = dataset.iloc[:, 10:14].values
y = dataset.iloc[:, 15:16].values

# Applying OneHotEncoder to some categorical columns
one_hot_encoder = OneHotEncoder(categories="auto")
season = one_hot_encoder.fit_transform(season).toarray()
year = one_hot_encoder.fit_transform(year).toarray()
month = one_hot_encoder.fit_transform(month).toarray()
hour = one_hot_encoder.fit_transform(hour).toarray()
holiday = one_hot_encoder.fit_transform(holiday).toarray()
weekday = one_hot_encoder.fit_transform(weekday).toarray()
workingday = one_hot_encoder.fit_transform(workingday).toarray()
weather = one_hot_encoder.fit_transform(weather).toarray()

# Concatenate Columns
X = np.concatenate((season, year, month, holiday, weekday, workingday, weather, A), axis=1)

# Train Test Splitting
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

# Applying Min Max Scaling
sc_X = MinMaxScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)
sc_y = MinMaxScaler()
y_train = sc_y.fit_transform(y_train)
y_test = sc_y.fit_transform(y_test)

# Define regressor by using Scikit-Learn
poly_reg = PolynomialFeatures(degree=2)

if R_TYPE == "random_forest":
    regressor = RandomForestRegressor(n_estimators=30, random_state=0)
elif R_TYPE == "decision_tree":
    regressor = DecisionTreeRegressor(random_state=0)
elif R_TYPE == "svm":
    regressor = SVR(kernel='rbf')
elif R_TYPE == "polynomial":
    x_poly = poly_reg.fit_transform(X_train)
    X_train = x_poly
    regressor = LinearRegression()
else:
    regressor = LinearRegression()

regressor.fit(X_train, y_train)

# Predict test data
if R_TYPE == "polynomial":
    y_predict = regressor.predict(poly_reg.fit_transform(X_test))
else:
    y_predict = regressor.predict(X_test)

# Calculate r_square, adjusted_r_square, root_mse, normal_root_mse and logarithmic_mse

r_square = r2_score(y_test, y_predict)
print('r_sqrt=', r_square)

adjusted_r_square = 1 - (1-r_square)*(len(y)-1)/(len(y)-X.shape[1]-1)
print('adj_r_sqrt=', adjusted_r_square)


root_mse = sqrt(mean_squared_error(y_test, y_predict))
print('root_mse=', root_mse)

normal_root_mse = root_mse/(max(y_test)-min(y_test))
print('normal_root_mse=', normal_root_mse)

logarithmic_mse = mean_squared_log_error(y_test, y_predict)
print('logarithmic_mse=', logarithmic_mse)

# Plot scatter plot of predicted and actual test data
plt.figure()
plt.scatter(y_test, y_predict, color='red')
qq = np.arange(0, max(y_predict), 0.001)
plt.plot(qq, qq, color='blue')
plt.title('Random Forest')
plt.xlabel('y_test')
plt.ylabel('y_predict')
plt.show()
