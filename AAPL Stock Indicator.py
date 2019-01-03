import pandas as pd
import numpy as np
import datetime
import quandl

from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

pd.set_option('display.expand_frame_repr', False)#allows wrap around if num cols exceeds max_cols

import getpass
USER_NAME = getpass.getuser()


def add_to_startup(file_path=""):
    if file_path == "":
        file_path = os.path.dirname(os.path.realpath(__file__))
    bat_path = r'C:\Users\%s\AppData\Roaming\Microsoft\Windows\Start Menu\Programs\Startup' % USER_NAME
    with open(bat_path + '\\' + "open.bat", "w+") as bat_file:
        bat_file.write(r'start "" %s' % file_path)


df = quandl.get('WIKI/AAPL')
df = df[['Adj. Close']]

forecast_out = int(30)
df['Prediction'] = df[['Adj. Close']].shift(-forecast_out)

X = np.array(df.drop(['Prediction'],1))
X = preprocessing.scale(X)

X_forecast = X[-forecast_out:]
X = X[:-forecast_out]
y = np.array(df['Prediction'])
y = y[:-forecast_out]


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2)

clf = LinearRegression()
clf.fit(X_train, y_train)

confidence = clf.score(X_test, y_test)
print('confidence: ', confidence)

forecast_prediction = clf.predict(X_forecast)
print(forecast_prediction)
