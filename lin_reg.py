import math

import pandas as pd
import quandl as qdl
import numpy as np
from sklearn import preprocessing, svm, model_selection
from sklearn.linear_model import  LinearRegression


# This is a sample Python script.

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.

def first_linReg():
    print("=¤" * 50)
    print()
    text = """
        Lest see some stock data!
        """
    print(text)
    print("=¤" * 50)
    text = """
        First step is to read the data, and keep it in a dataframe:
        df = qd.get('')
        """
    print(text)
    df = qdl.get('WIKI/GOOGL', api_key='oQYzTPV13ZWK2tyU_TKc')
    print(df)
    print("=¤" * 50)
    text = """
        Second step: redefine our dataset only with required columns:
        df = df[['Adj. Open','Adj. High','Adj. Low','Adj. Close','Adj. Volume']]
        """
    print(text)
    print("=¤" * 50)
    df = df[['Adj. Open', 'Adj. High', 'Adj. Low', 'Adj. Close', 'Adj. Volume']]
    text = """
        Third step : define additional columns, to illustrate important changes:
        df['HL_PCT'] = (df['Adj. High']-df['Adj. Low'])/df['Adj. Close']
        """
    print(text)
    df['HL_PCT'] = (df['Adj. High'] - df['Adj. Low']) / df['Adj. Close'] * 100.0
    print("=¤" * 50)
    text = """
        Fourth step : define additional column - 2 ,daily movements:
        df['PCT_change'] = (df['Adj. Close']-df['Adj. Open'])/df['Adj. Open']*100.0
        """
    print(text)
    df['PCT_change'] = (df['Adj. Close'] - df['Adj. Open']) / df['Adj. Close'] * 100.0
    print("=¤" * 50)
    text = """
        Fifth step : re-define the dataset again with necessary columns:
        df = df[['Adj. Close','HL_PCT','PCT_change'.'Adj. Volume']]*100.0
        """
    print(text)
    df = df[['Adj. Close', 'HL_PCT', 'PCT_change', 'Adj. Volume']]
    print(df.head())
    print("=¤" * 50)
    text = """
        Sixth step : now have the data, now we have to consider, that what we would like to PREDICT.
        For this we will add a new column_: forecast_col
        forecast_col = 'Adj. Close'
        Seventh step: now we should clear the data we have. For example check fro missing data, and
        fill it up with some value, to not waste other columns (dropna/fillna)
        df.fillna(-99999,inplace=True)
        """
    print(text)
    forecast_col = 'Adj. Close'
    df.fillna(-99999, inplace=True)
    print("=¤" * 50)
    text = """
        Eigth step : now have define a new variable for the forecast value:
        forecast_out = int(math.ceil(0.1*len(df)))
        This will represent the 10% of the dataframe timespan
        """
    print(text)
    forecast_out = int(math.ceil(0.01 * len(df)))
    print("=¤" * 50)
    text = """
        Ninth step : now add a new column with the value, and shift it up with the forecast_out amount
        df['label'] = df[forecast_col].shift(-forecast_out)
        """
    print(text)
    df['label'] = df[forecast_col].shift(-forecast_out)
    # %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    print("=¤" * 50)
    text = """
        Additional step 001: for sake of security lest write the data into a csv file:
        df.to_csv(r'F:\\F_Dokumentumok\\PYTHON\\DATA\\WKI_GOOGL.csv')
        """
    print(text)
    df.to_csv(r'F:\F_Dokumentumok\PYTHON\DATA\WKI_GOOGL.csv')
    df_res = pd.read_csv(r'F:\F_Dokumentumok\PYTHON\DATA\WKI_GOOGL.csv')
    print("df_res.head(5):")
    print(df_res.head(5))
    print("=¤" * 50)
    text = """
        Lest see how linear regression works in sklearn!
        First we need to variable vectors:
        x = np.array(df.drop(['label'],1))
        y = np.array(df['label'])
        Then we need the scales to have statistically correct sets
        X = preprocessing.scale(X)
        Now we redefine X as the one row higher value
        X = X[:-forecast_out+1]
        Now lets throw away the NaN values
        df=df.dropna(inplace=True)
        Now define our Y values as well
        y = np.array(df['label'])
        And lets check its length with a print...
        print("len(X):",len(X),", len(y):",len(y))
        """
    print(text)
    df.dropna(inplace=True)
    x = np.array(df.drop(['label'], 1))
    y = np.array(df['label'])
    X = preprocessing.scale(x)
    y = np.array(df['label'])
    print("len(X):", len(X), ", len(y):", len(y))
    print("=¤" * 50)
    text = """
        Now we are ready to test our training set:
        X_train, X_test,  y_train, y_test = model_selection.train_test_split(X,y,test_size=0.2)
        Now lets define a classifier with the model Linearregression
        clf = LinearRegression()
        Now run the fit method of the classifier
        clf.fit(X_train,y_train)
        And a the end lest see what kjind of "score" the classifier can achieve
        accuracy = clf.score(X_test,y_test)
        print("Oir current accuracy is: ", accuracy)
        """
    print(text)
    X_train, X_test, y_train, y_test = model_selection.train_test_split(X, y, test_size=0.2)
    clf = LinearRegression()
    clf.fit(X_train, y_train)
    accuracy = clf.score(X_test, y_test)
    print("Our current accuracy is: ", accuracy)
    print("=¤" * 50)
    text = """
        Now lets see a different algorithm:
        print(text)
        X_train, X_test, y_train, y_test = model_selection.train_test_split(X, y, test_size=0.2)
        clf = svm.SVR()
        clf.fit(X_train, y_train)
        accuracy = clf.score(X_test, y_test)
        print("Our current SVR-accuracy is: ", accuracy)
        """
    print(text)
    X_train, X_test, y_train, y_test = model_selection.train_test_split(X, y, test_size=0.2)
    clf = svm.SVR(kernel='linear')
    clf.fit(X_train, y_train)
    svr_accuracy = clf.score(X_test, y_test)
    print("Our current SVR-accuracy is: ", svr_accuracy)
    print("=¤" * 50)


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    first_linReg()


