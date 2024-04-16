import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import MinMaxScaler
import pickle
import os

def load_and_preprocess_data(filepath):
    data = pd.read_csv(/content/sample_data/nigeria_houses_data.csv)
    
    # Handling missing values - impute with median
    for column in ['bedrooms', 'bathrooms', 'toilets', 'parking_space']:
        data[column].fillna(data[column].median(), inplace=True)
    
    # Remove outliers
    Q1 = data['price'].quantile(0.25)
    Q3 = data['price'].quantile(0.75)
    IQR = Q3 - Q1
    data = data[~((data['price'] < (Q1 - 1.5 * IQR)) | (data['price'] > (Q3 + 1.5 * IQR)))]
    
    # Feature and target separation
    X = data[['bedrooms', 'bathrooms', 'toilets', 'parking_space']]  # Assuming these are the features
    y = data['price']
    
    # Data scaling
    scaler = MinMaxScaler()
    X_scaled = scaler.fit_transform(X)
    
    return X_scaled, y

def train_model(X, y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    
    # Predict on the test set
    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    print(f"Test MSE: {mse}")

    return model

def save_model(model, model_path):
    with open(model_path, 'wb') as file:
        pickle.dump(model, file)

if __name__ == "__main__":
    # Adjust the filepath accordingly
    filepath = '/content/sample_data/nigeria_houses_data.csv'
    model_path = 'saved_models/house_pricing_model.pkl'
    
    X, y = load_and_preprocess_data(filepath)
    model = train_model(X, y)
    save_model(model, model_path)
    print("Model training complete and model has been saved.")
