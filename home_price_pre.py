import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Load Dataset
data = pd.read_csv('home_prices.csv')  # Replace with actual dataset path

# Display first few rows
data.head()

# Preprocess Data
# Handling missing values
data.fillna(data.mean(), inplace=True)

# Convert categorical features to numerical
categorical_cols = ['location', 'neighborhood']  # Adjust as per dataset
le = LabelEncoder()
for col in categorical_cols:
    data[col] = le.fit_transform(data[col])

# Feature Selection
features = ['income', 'schools', 'hospitals', 'crime_rate', 'location', 'neighborhood']  # Adjust columns
X = data[features]
y = data['price']

# Split Data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Normalize Features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Train XGBoost Model
model = xgb.XGBRegressor(objective='reg:squarederror', n_estimators=100, learning_rate=0.1, max_depth=5)
model.fit(X_train, y_train)

# Predictions
y_pred = model.predict(X_test)

# Model Evaluation
mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, y_pred)

print(f'MAE: {mae}')
print(f'RMSE: {rmse}')
print(f'R2 Score: {r2}')

# Hyperparameter Tuning
param_grid = {
    'n_estimators': [100, 200],
    'max_depth': [3, 5, 7],
    'learning_rate': [0.01, 0.1, 0.2]
}

grid_search = GridSearchCV(model, param_grid, cv=3, scoring='r2', verbose=2, n_jobs=-1)
grid_search.fit(X_train, y_train)

print("Best Parameters:", grid_search.best_params_)

# Final Model with Best Params
best_model = grid_search.best_estimator_
y_pred_best = best_model.predict(X_test)

# Final Evaluation
print(f'Final R2 Score: {r2_score(y_test, y_pred_best)}')
