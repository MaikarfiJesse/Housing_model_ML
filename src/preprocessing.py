import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

def load_data(filepath):
    data = pd.read_csv('/content/sample_data/nigeria_houses_data.csv')
    return data

def clean_data(data):
    # Handling missing values
    data.fillna(data.median(), inplace=True)
    # Handling outliers in price
    Q1 = data['price'].quantile(0.25)
    Q3 = data['price'].quantile(0.75)
    IQR = Q3 - Q1
    data['price'] = np.where(data['price'] < (Q1 - 1.5 * IQR), Q1 - 1.5 * IQR, data['price'])
    data['price'] = np.where(data['price'] > (Q3 + 1.5 * IQR), Q3 + 1.5 * IQR, data['price'])
    return data

def preprocess_features(data):
    # Assuming 'price' is the last column
    X = data.iloc[:, :-1]
    y = data.iloc[:, -1]
    scaler = MinMaxScaler()
    X_scaled = scaler.fit_transform(X)
    return X_scaled, y

def split_data(X, y):
    X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.3, random_state=42)
    X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)
    return X_train, X_val, X_test, y_train, y_val, y_test
