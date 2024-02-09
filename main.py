from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import LabelEncoder
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from sklearn.metrics import r2_score

# Učitavanje podataka i prethodni koraci preprocesiranja
data = pd.read_csv('diamonds.csv')
data.drop(data.columns[0], axis=1, inplace=True)

y = data['price']
X = data.drop('price', axis=1)

encoder = LabelEncoder()

X['cut'] = encoder.fit_transform(X['cut'])
cut_mappings = {index: label for index, label in enumerate(encoder.classes_)}

X['color'] = encoder.fit_transform(X['color'])
color_mappings = {index: label for index, label in enumerate(encoder.classes_)}

X['clarity'] = encoder.fit_transform(X['clarity'])
clarity_mappings = {index: label for index, label in enumerate(encoder.classes_)}

scaler = MinMaxScaler()
X = scaler.fit_transform(X)

# Podela podataka na trening i test skupove
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Linear regression
std_model = LinearRegression()
std_model.fit(X_train, y_train)
print(f'---Without regularization: {std_model.score(X_test, y_test)}')


# Decision Tree Regressor
dt_model = DecisionTreeRegressor()
dt_model.fit(X_train, y_train)
dt_predictions = dt_model.predict(X_test)
dt_r2 = r2_score(y_test, dt_predictions)
print(f"Decision Tree R^2 Score: {dt_r2}")

# Random Forest Regressor
rf_model = RandomForestRegressor()
rf_model.fit(X_train, y_train)
rf_predictions = rf_model.predict(X_test)
rf_r2 = r2_score(y_test, rf_predictions)
print(f"Random Forest R^2 Score: {rf_r2}")

# Support Vector Regression
svr_model = SVR()
svr_model.fit(X_train, y_train)
svr_predictions = svr_model.predict(X_test)
svr_r2 = r2_score(y_test, svr_predictions)
print(f"SVR R^2 Score: {svr_r2}")
