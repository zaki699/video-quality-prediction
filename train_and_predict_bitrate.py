import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from xgboost import XGBRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge, Lasso
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import cross_val_score

# Step 1: Load your dataset (example)
data = pd.read_csv('video_quality_data.csv')

# Step 2: Split the data into features (X) and target labels (y)
X = data[['Scene Complexity', 'Resolution (px)', 'Frame Rate (fps)', 'CRF', 'average_framerate', 'min_framerate', 'max_framerate', 'smoothed_frame_rate_variation']].values
y_bitrate = data['bitrate'].values

# Step 3: Split the data into training and testing sets for bitrate prediction
X_train, X_test, y_bitrate_train, y_bitrate_test = train_test_split(X, y_bitrate, test_size=0.2, random_state=42)

# Step 4: Create models with L1 (Lasso), L2 (Ridge) regularization, and Random Forest or XGBoost for bitrate prediction
models = {
    'Lasso': Pipeline([('scaler', StandardScaler()), ('model', Lasso(alpha=0.1))]),
    'Ridge': Pipeline([('scaler', StandardScaler()), ('model', Ridge(alpha=1.0))]),
    'RandomForest': RandomForestRegressor(n_estimators=100, random_state=42),
    'XGBoost': XGBRegressor(objective='reg:squarederror', n_estimators=100, random_state=42)
}

# Step 5: Train and evaluate each model for bitrate prediction
for name, model in models.items():
    print(f"Training {name} model for bitrate prediction...")
    model.fit(X_train, y_bitrate_train)
    
    # Predict on the test set
    bitrate_predictions = model.predict(X_test)
    
    # Calculate the Mean Squared Error (MSE)
    mse = mean_squared_error(y_bitrate_test, bitrate_predictions)
    print(f"{name} Model Bitrate MSE: {mse:.4f}")
    
    # Perform cross-validation
    cv_scores = cross_val_score(model, X_train, y_bitrate_train, cv=5, scoring='neg_mean_squared_error')
    print(f"{name} Model Cross-Validation MSE: {-cv_scores.mean():.4f}\n")

# Feature importance from XGBoost
xgb_model = models['XGBoost']
xgb_feature_importances = xgb_model.feature_importances_
print("XGBoost Feature Importances:")
for feature, importance in zip(['Scene Complexity', 'Resolution (px)', 'Frame Rate (fps)', 'CRF', 'average_framerate', 'min_framerate', 'max_framerate', 'smoothed_frame_rate_variation'], xgb_feature_importances):
    print(f"{feature}: {importance:.4f}")
