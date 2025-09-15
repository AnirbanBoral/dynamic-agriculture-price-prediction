# -*- coding: utf-8 -*-
"""
Created on Sat Aug 30 19:09:49 2025

@author: Anirban Boral
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import OneHotEncoder
from sklearn.linear_model import LinearRegression


# Load dataset
dataset = pd.read_csv('ProjectData.csv')

# Process 'Arrival_Date' - convert to datetime and then to ordinal (numeric)
dataset['Arrival_Date'] = pd.to_datetime(dataset['Arrival_Date'], errors='coerce')
dataset['Arrival_Ordinal'] = dataset['Arrival_Date'].map(lambda x: x.toordinal() if pd.notnull(x) else 0)

# Select categorical columns to encode
categorical_cols = ["Market", "Commodity", "Variety", "Grade"]
encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
encoded_features = encoder.fit_transform(dataset[categorical_cols])

# Numeric features for concatenation (must be 2D arrays)
numeric_features = dataset[['Arrival_Ordinal', 'Commodity_Code']].fillna(0).values

# Concatenate encoded categorical and numeric features horizontally
X = np.hstack([encoded_features, numeric_features])

# Target variable
y = dataset["Modal_Price"]

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Fit linear regression
model = LinearRegression()
model.fit(X_train, y_train)

# Predict and plot actual vs predicted
y_pred = model.predict(X_test)
plt.scatter(y_test, y_pred)
plt.xlabel('Actual Modal Price')
plt.ylabel('Predicted Modal Price')
plt.title('Linear Regression Prediction')
plt.show()

"""
dataset=pd.read_csv('ProjectData.csv')
print(dataset['Min_Price'].head(20))
print(dataset['Min_Price'].head(20))
features=["Market","Commodity","Variety","Grade","Arrival_Date","Commodity_Code"]
X_train,X_test,y_train,y_test=train_test_split(dataset[features],dataset["Modal_Price"],test_size=0.2,random_state=42)
plt.plot(X_test,y_test)"""