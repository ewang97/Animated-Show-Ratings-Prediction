
import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from sklearn.model_selection import cross_val_predict
from sklearn import linear_model
from sklearn.linear_model import LinearRegression,LogisticRegression
from sklearn.ensemble import RandomForestRegressor


from sklearn import metrics
from sklearn.metrics import mean_squared_error
import chardet

def preprocess_data(data = 'data/Animated_Tv_Series.csv'):
    #format with proper encoding of data file
    with open(data, 'rb') as f:
        result = chardet.detect(f.read())
    df = pd.read_csv(data, encoding=result['encoding'])

    #Cleaning/Feature Engineering for Initial Runtime and Animation styles

    techn = df['Technique'].unique()
    arr_ = list(map(lambda x: len(x), techn)) 
    x = techn[np.argsort(arr_)]

    one_hot_cols_technique = ["CGI","Flash","Traditional","Stop","Live","Digital"]

    for col in one_hot_cols_technique:
        df[f"{col}"] = np.where(df['Technique'].str.contains(f"{col}"),1,0)

    df['startEnd'] = df['Year'].str.split("-|,")

    df["initRuntime"] = [1 if len(startend) == 1 else int(startend[1]) - int(startend[0]) if not "present" in startend[1] else -1 for startend in df['startEnd']]

    df["Google users"] = df['Google users'].str.rstrip('%').astype('float') / 100.0

    features = ['Episodes', 'Google users', 'CGI','Flash','Traditional','Stop','Live','Digital','initRuntime','IMDb']
    cleaned_df = df[~df[features].isnull().any(axis=1)]
    cleaned_df = cleaned_df.iloc[np.where(cleaned_df['initRuntime'] != -1)]

    return cleaned_df

def data_split(df, test_size = 0.3):
    X_features = ['Episodes', 'Google users', 'CGI','Flash','Traditional','Stop','Live','Digital','initRuntime']
    X = df[X_features]
    y = df['IMDb']

    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size)

    return X_train, X_test, y_train, y_test

def train_model(X_train, X_test, y_train, y_test, linear = True):

    if linear:
        linear_m = LinearRegression().fit(X_train, y_train)

        #heteroskedasticity, need to transform some variable
        # residuals = y_test-lin_pred
        # plt.scatter(y_test,lin_pred)

        # plt.show()

        forest = RandomForestRegressor(300)
        rf = forest.fit(X_train, y_train)

        yfit = forest.predict(X_test)
        #MSE

        # plt.scatter(y_test,yfit)

        # plt.show()
        model = linear_m
    else:
        forest = RandomForestRegressor(300)
        forest.fit(X_train, y_train)
        model = forest
    return model