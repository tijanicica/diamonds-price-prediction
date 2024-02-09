from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import LabelEncoder
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_squared_error

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

# Convert X back to a pandas DataFrame
X_df = pd.DataFrame(X, columns=data.drop('price', axis=1).columns)

# Calculate correlation matrix
correlation_matrix = X_df.corr()

# Plot heatmap
plt.figure(figsize=(10, 8))
sns.heatmap(correlation_matrix, annot=True, cmap='YlOrRd', fmt=".2f", linewidths=0.5)
plt.title('Correlation Matrix')
plt.show()

# Drop columns 'x', 'y', and 'z' from the DataFrame
#v iskljucujemo one koji su u velikoj korelaciji
X_df.drop(['x', 'y', 'z'], axis=1, inplace=True)

# Convert DataFrame back to NumPy array
X = X_df.values

# Podela podataka na trening i test skupove
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Linear regression
std_model = LinearRegression()
std_model.fit(X_train, y_train)
print(f'Linear Regression R^2 Score: {std_model.score(X_test, y_test)}')

# Predictions
y_pred = std_model.predict(X_test)

# Residuals calculation
residuals = y_test - y_pred

# Assumption 2: Independence of Residuals // Nezavisnost gresaka
plt.scatter(y_pred, residuals, color='orange')
plt.xlabel('Predicted values')
plt.ylabel('Residuals')
plt.title('Residuals vs Predicted values')
plt.show()

# Assumption 3: Homoscedasticity // Konstantna varijansa gresaka
plt.scatter(y_pred, residuals, color='orange')
plt.axhline(y=0, color='r', linestyle='-')
plt.xlabel('Predicted values')
plt.ylabel('Residuals')
plt.title('Residuals vs Predicted values')
plt.show()

# Assumption 4: Normality of Residuals // Normalnost gresaka
sns.histplot(residuals, kde=True, color='orange')
plt.xlabel('Residuals')
plt.ylabel('Frequency')
plt.title('Residuals Distribution')
plt.show()

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

def min_price_with_carat_and_cut(data, min_carat, cut_type='Ideal'):

    filtered_data = data[(data['carat'] > min_carat) & (data['cut'] == cut_type)]
    if len(filtered_data) == 0:
        return f"No diamonds found with carat > {min_carat} and cut type '{cut_type}'."
    return filtered_data['price'].min()

min_price_carat_and_cut = min_price_with_carat_and_cut(data, 0.5, 'Ideal')
print(f"Minimum price for diamonds with carat > 0.5 and 'Ideal' cut: ${min_price_carat_and_cut:.2f}")


def max_price_with_carat_and_cut(data, min_carat, cut_type='Ideal'):

    filtered_data = data[(data['carat'] > min_carat) & (data['cut'] == cut_type)]
    if len(filtered_data) == 0:
        return f"No diamonds found with carat > {min_carat} and cut type '{cut_type}'."
    return filtered_data['price'].max()

max_price_carat_and_cut = max_price_with_carat_and_cut(data, 0.5, 'Ideal')
print(f"Maximum price for diamonds with carat > 0.5 and 'Ideal' cut: ${max_price_carat_and_cut:.2f}")


def min_price_with_carat_and_cut_affordable(data, max_carat, cut_type='Fair'):

    filtered_data = data[(data['carat'] < max_carat) & (data['cut'] == cut_type)]
    if len(filtered_data) == 0:
        return f"No diamonds found with carat < {max_carat} and cut type '{cut_type}'."
    return filtered_data['price'].min()

# Test the function
min_price_carat_and_cut = min_price_with_carat_and_cut_affordable(data, 0.3, 'Fair')
print(f"Minimum price for diamonds with carat < 0.3 and 'Fair' cut: ${min_price_carat_and_cut:.2f}")


def max_price_with_carat_and_cut_affordable(data, max_carat, cut_type='Fair'):

    filtered_data = data[(data['carat'] < max_carat) & (data['cut'] == cut_type)]
    if len(filtered_data) == 0:
        return f"No diamonds found with carat < {max_carat} and cut type '{cut_type}'."
    return filtered_data['price'].max()

max_price_carat_and_cut = max_price_with_carat_and_cut_affordable(data, 0.3, 'Fair')
print(f"Maximum price for diamonds with carat < 0.3 and 'Fair' cut: ${max_price_carat_and_cut:.2f}")


