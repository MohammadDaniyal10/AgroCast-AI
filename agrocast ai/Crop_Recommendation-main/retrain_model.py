import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.ensemble import RandomForestClassifier
import pickle

# Load data
crop = pd.read_csv("Crop_recommendation.csv")

# Create label mapping
crop_dict = {
    'rice': 1, 'maize': 2, 'jute': 3, 'cotton': 4, 'coconut': 5,
    'papaya': 6, 'orange': 7, 'apple': 8, 'muskmelon': 9, 'watermelon': 10,
    'grapes': 11, 'mango': 12, 'banana': 13, 'pomegranate': 14, 'lentil': 15,
    'blackgram': 16, 'mungbean': 17, 'mothbeans': 18, 'pigeonpeas': 19,
    'kidneybeans': 20, 'chickpea': 21, 'coffee': 22
}

# Map labels to numbers
crop['label'] = crop['label'].map(crop_dict)

# Split features and target
X = crop.drop('label', axis=1)
y = crop['label']

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Scale the features
mx = MinMaxScaler()
X_train_mx = mx.fit_transform(X_train)
X_test_mx = mx.transform(X_test)

sc = StandardScaler()
X_train_scaled = sc.fit_transform(X_train_mx)
X_test_scaled = sc.transform(X_test_mx)

# Train model
model = RandomForestClassifier(random_state=42)
model.fit(X_train_scaled, y_train)

# Save model and scalers
pickle.dump(model, open('model.pkl', 'wb'))
pickle.dump(mx, open('minmaxscaler.pkl', 'wb'))
pickle.dump(sc, open('standscaler.pkl', 'wb'))

print("Model training completed and saved successfully!")