# program for testing rf_wrapper.py
import os
import pandas as pd
#from Modules.rf_wrapper import RF_Model, Preprocess
from rf_wrapper import RF_Model, Preprocess

# set up directories
data_dir = os.getcwd() + r'/Data'
model_dir = os.getcwd() + r'/Models'

# load testing data
X = pd.read_csv((data_dir + '/x_test.csv'), nrows=5)
y = pd.read_csv((data_dir + '/y_test.csv'), nrows=5)

# load model and predict on test data
model = RF_Model()

if model.LoadGridSearch(f'{model_dir}/random_forest.joblib'):
    print("Successfully loaded model.")

if model.LoadScaler(f'{model_dir}/std_scaler.joblib'):
    print("Successfully loaded scaler.")
    
    
pred = model.Predict(X, is_scaled=True)

print(f"{'Idx':<5} | {'Predicted':<13} | Truth")
for idx in enumerate(pred):
    print(f"{idx[0]:<5n} | {pred[idx[0]]:<13} | {y[' Label'].iloc[idx[0]]}")