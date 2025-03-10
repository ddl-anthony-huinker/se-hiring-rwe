#!/usr/bin/env python
# Tabular Data: Neural Network vs. Traditional ML Comparison

# Required packages
# pip install tensorflow scikit-learn pandas matplotlib numpy seaborn

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import time
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.utils import plot_model

# 1. Load and prepare the Wine Quality dataset
def load_and_prepare_data():
    # Download the dataset if not already present
    red_wine_url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-red.csv'
    white_wine_url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-white.csv'
    
    if not os.path.exists('winequality-red.csv'):
        os.system(f'wget {red_wine_url}')
    if not os.path.exists('winequality-white.csv'):
        os.system(f'wget {white_wine_url}')
    
    # Load datasets
    red_wine = pd.read_csv('winequality-red.csv', sep=';')
    white_wine = pd.read_csv('winequality-white.csv', sep=';')
    
    # Add a type column to distinguish red and white wines
    red_wine['wine_type'] = 1  # Red wine
    white_wine['wine_type'] = 0  # White wine
    
    # Combine datasets
    wines = pd.concat([red_wine, white_wine], axis=0)
    
    # Display basic information
    print("Dataset Shape:", wines.shape)
    print("\nFeature Statistics:")
    print(wines.describe())
    
    # Check for missing values
    print("\nMissing Values:", wines.isnull().sum().sum())
    
    # Separate features and target
    X = wines.drop('quality', axis=1)
    y = wines['quality']
    
    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    # Scale the features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    return X_train, X_test, X_train_scaled, X_test_scaled, y_train, y_test, wines

# 2. Build and train Neural Network model
def build_and_train_nn(X_train, y_train, X_test, y_test):
    print("Training Neural Network model...")
    start_time = time.time()
    
    # Build model
    model = Sequential([
        Dense(64, activation='relu', input_shape=(X_train.shape[1],)),
        Dropout(0.2),
        Dense(32, activation='relu'),
        Dropout(0.2),
        Dense(16, activation='relu'),
        Dense(1)  # Linear activation for regression
    ])
    
    # Compile model
    model.compile(optimizer=Adam(learning_rate=0.001), loss='mse', metrics=['mae'])
    
    # Set up early stopping
    early_stopping = EarlyStopping(
        monitor='val_loss',
        patience=10,
        restore_best_weights=True
    )
    
    # Train model
    history = model.fit(
        X_train, y_train,
        epochs=100,
        batch_size=32,
        validation_split=0.2,
        callbacks=[early_stopping],
        verbose=1
    )
    
    # Evaluate model
    nn_train_time = time.time() - start_time
    print(f"Neural Network training time: {nn_train_time:.2f} seconds")
    
    start_time = time.time()
    y_pred_nn = model.predict(X_test)
    nn_inference_time = time.time() - start_time
    
    # Calculate metrics
    nn_mse = mean_squared_error(y_test, y_pred_nn)
    nn_rmse = np.sqrt(nn_mse)
    nn_mae = mean_absolute_error(y_test, y_pred_nn)
    nn_r2 = r2_score(y_test, y_pred_nn)
    
    print(f"Neural Network inference time: {nn_inference_time:.2f} seconds")
    print(f"Neural Network RMSE: {nn_rmse:.4f}")
    print(f"Neural Network MAE: {nn_mae:.4f}")
    print(f"Neural Network R²: {nn_r2:.4f}")
    
    return model, history, nn_rmse, nn_mae, nn_r2, nn_train_time, nn_inference_time, y_pred_nn

# 3. Train and evaluate Gradient Boosting Regressor
def train_gradient_boosting(X_train, y_train, X_test, y_test):
    print("Training Gradient Boosting model...")
    start_time = time.time()
    
    # Train model
    gb = GradientBoostingRegressor(
        n_estimators=200,
        max_depth=5,
        learning_rate=0.1,
        random_state=42
    )
    gb.fit(X_train, y_train)
    
    gb_train_time = time.time() - start_time
    print(f"Gradient Boosting training time: {gb_train_time:.2f} seconds")
    
    # Predict and measure inference time
    start_time = time.time()
    y_pred_gb = gb.predict(X_test)
    gb_inference_time = time.time() - start_time
    
    # Calculate metrics
    gb_mse = mean_squared_error(y_test, y_pred_gb)
    gb_rmse = np.sqrt(gb_mse)
    gb_mae = mean_absolute_error(y_test, y_pred_gb)
    gb_r2 = r2_score(y_test, y_pred_gb)
    
    print(f"Gradient Boosting inference time: {gb_inference_time:.2f} seconds")
    print(f"Gradient Boosting RMSE: {gb_rmse:.4f}")
    print(f"Gradient Boosting MAE: {gb_mae:.4f}")
    print(f"Gradient Boosting R²: {gb_r2:.4f}")
    
    # Feature importance
    feature_imp = pd.DataFrame(
        sorted(zip(gb.feature_importances_, X_train.columns)),
        columns=['Value', 'Feature']
    )
    
    return gb, gb_rmse, gb_mae, gb_r2, gb_train_time, gb_inference_time, feature_imp, y_pred_gb

# 4. Train and evaluate Random Forest Regressor
def train_random_forest(X_train, y_train, X_test, y_test):
    print("Training Random Forest model...")
    start_time = time.time()
    
    # Train model
    rf = RandomForestRegressor(
        n_estimators=200,
        max_depth=10,
        random_state=42
    )
    rf.fit(X_train, y_train)
    
    rf_train_time = time.time() - start_time
    print(f"Random Forest training time: {rf_train_time:.2f} seconds")
    
    # Predict and measure inference time
    start_time = time.time()
    y_pred_rf = rf.predict(X_test)
    rf_inference_time = time.time() - start_time
    
    # Calculate metrics
    rf_mse = mean_squared_error(y_test, y_pred_rf)
    rf_rmse = np.sqrt(rf_mse)
    rf_mae = mean_absolute_error(y_test, y_pred_rf)
    rf_r2 = r2_score(y_test, y_pred_rf)
    
    print(f"Random Forest inference time: {rf_inference_time:.2f} seconds")
    print(f"Random Forest RMSE: {rf_rmse:.4f}")
    print(f"Random Forest MAE: {rf_mae:.4f}")
    print(f"Random Forest R²: {rf_r2:.4f}")
    
    return rf, rf_rmse, rf_mae, rf_r2, rf_train_time, rf_inference_time, y_pred_rf

# 5. Plot learning curves and results
def plot_results(history, nn_rmse, gb_rmse, rf_rmse, nn_mae, gb_mae, rf_mae, 
                nn_r2, gb_r2, rf_r2, nn_train_time, gb_train_time, rf_train_time,
                nn_inference_time, gb_inference_time, rf_inference_time,
                feature_imp, wines, y_test, y_pred_nn, y_pred_gb, y_pred_rf):
    # Plot NN learning curves
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    plt.plot(history.history['loss'], label='Train Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title('Neural Network Training Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss (MSE)')
    plt.legend()
    
    plt.subplot(1, 2, 2)
    plt.plot(history.history['mae'], label='Train MAE')
    plt.plot(history.history['val_mae'], label='Validation MAE')
    plt.title('Neural Network Training MAE')
    plt.xlabel('Epoch')
    plt.ylabel('MAE')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig('nn_learning_curves.png')
    
    # Plot feature importance for Gradient Boosting
    plt.figure(figsize=(10, 6))
    sns.barplot(x='Value', y='Feature', data=feature_imp.sort_values(by='Value', ascending=False))
    plt.title('Feature Importance (Gradient Boosting)')
    plt.tight_layout()
    plt.savefig('feature_importance.png')
    
    # Plot error metrics comparison
    plt.figure(figsize=(15, 5))
    
    # RMSE comparison
    plt.subplot(1, 3, 1)
    models = ['Neural Network', 'Gradient Boosting', 'Random Forest']
    rmse_values = [nn_rmse, gb_rmse, rf_rmse]
    plt.bar(models, rmse_values, color=['blue', 'green', 'orange'])
    plt.title('RMSE Comparison')
    plt.ylabel('RMSE')
    plt.xticks(rotation=45)
    
    # MAE comparison
    plt.subplot(1, 3, 2)
    mae_values = [nn_mae, gb_mae, rf_mae]
    plt.bar(models, mae_values, color=['blue', 'green', 'orange'])
    plt.title('MAE Comparison')
    plt.ylabel('MAE')
    plt.xticks(rotation=45)
    
    # R² comparison
    plt.subplot(1, 3, 3)
    r2_values = [nn_r2, gb_r2, rf_r2]
    plt.bar(models, r2_values, color=['blue', 'green', 'orange'])
    plt.title('R² Comparison')
    plt.ylabel('R²')
    plt.xticks(rotation=45)
    
    plt.tight_layout()
    plt.savefig('error_metrics_comparison.png')
    
    # Time comparison
    plt.figure(figsize=(12, 5))
    
    x = np.arange(len(models))
    width = 0.35
    
    plt.subplot(1, 2, 1)
    plt.bar(x, [nn_train_time, gb_train_time, rf_train_time], width)
    plt.xticks(x, models, rotation=45)
    plt.title('Training Time Comparison')
    plt.ylabel('Time (seconds)')
    
    plt.subplot(1, 2, 2)
    plt.bar(x, [nn_inference_time, gb_inference_time, rf_inference_time], width)
    plt.xticks(x, models, rotation=45)
    plt.title('Inference Time Comparison')
    plt.ylabel('Time (seconds)')
    
    plt.tight_layout()
    plt.savefig('time_comparison.png')
    
    # Actual vs Predicted plots
    plt.figure(figsize=(15, 5))
    
    plt.subplot(1, 3, 1)
    plt.scatter(y_test, y_pred_nn, alpha=0.5)
    plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], 'r--')
    plt.title('Neural Network: Actual vs Predicted')
    plt.xlabel('Actual')
    plt.ylabel('Predicted')
    
    plt.subplot(1, 3, 2)
    plt.scatter(y_test, y_pred_gb, alpha=0.5)
    plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], 'r--')
    plt.title('Gradient Boosting: Actual vs Predicted')
    plt.xlabel('Actual')
    plt.ylabel('Predicted')
    
    plt.subplot(1, 3, 3)
    plt.scatter(y_test, y_pred_rf, alpha=0.5)
    plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], 'r--')
    plt.title('Random Forest: Actual vs Predicted')
    plt.xlabel('Actual')
    plt.ylabel('Predicted')
    
    plt.tight_layout()
    plt.savefig('actual_vs_predicted.png')

# 6. Main function to run the experiment
def main():
    # Load and prepare data
    X_train, X_test, X_train_scaled, X_test_scaled, y_train, y_test, wines = load_and_prepare_data()
    
    # Train and evaluate Neural Network
    nn_model, history, nn_rmse, nn_mae, nn_r2, nn_train_time, nn_inference_time, y_pred_nn = build_and_train_nn(
        X_train_scaled, y_train, X_test_scaled, y_test
    )
    
    # Train and evaluate Gradient Boosting
    gb_model, gb_rmse, gb_mae, gb_r2, gb_train_time, gb_inference_time, feature_imp, y_pred_gb = train_gradient_boosting(
        X_train, y_train, X_test, y_test
    )
    
    # Train and evaluate Random Forest
    rf_model, rf_rmse, rf_mae, rf_r2, rf_train_time, rf_inference_time, y_pred_rf = train_random_forest(
        X_train, y_train, X_test, y_test
    )
    
    # Plot results
    plot_results(
        history, nn_rmse, gb_rmse, rf_rmse, nn_mae, gb_mae, rf_mae,
        nn_r2, gb_r2, rf_r2, nn_train_time, gb_train_time, rf_train_time,
        nn_inference_time, gb_inference_time, rf_inference_time,
        feature_imp, wines, y_test, y_pred_nn, y_pred_gb, y_pred_rf
    )
    
    # Create summary table
    summary = pd.DataFrame({
        'Model': ['Neural Network', 'Gradient Boosting', 'Random Forest'],
        'RMSE': [nn_rmse, gb_rmse, rf_rmse],
        'MAE': [nn_mae, gb_mae, rf_mae],
        'R²': [nn_r2, gb_r2, rf_r2],
        'Training Time (s)': [nn_train_time, gb_train_time, rf_train_time],
        'Inference Time (s)': [nn_inference_time, gb_inference_time, rf_inference_time]
    })
    
    print("\nModel Comparison Summary:")
    print(summary)
    summary.to_csv('tabular_model_comparison.csv', index=False)

if __name__ == "__main__":
    main()