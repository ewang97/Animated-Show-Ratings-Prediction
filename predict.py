
import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from sklearn.model_selection import cross_val_predict
from sklearn.linear_model import LogisticRegression

from sklearn import metrics
from sklearn.metrics import mean_squared_error
from train import preprocess_data,data_split,train_model
import chardet
import matplotlib.pyplot as plt

def predict(model, X_test):
    return model.predict(X_test)

    
if __name__ == "__main__":
    full_df = preprocess_data()

    X_train, X_test, y_train, y_test = data_split(full_df,0.3)
    model = train_model(X_train, X_test, y_train, y_test)
    predictions = predict(model, X_test)
    mse = mean_squared_error(y_test, predictions)
    print('Success - MSE is: ' + str(mse))