import pandas as pd
import numpy as np
import quandl
import matplotlib.pyplot as plt

from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

pd.set_option('display.expand_frame_repr', False)#allows wrap around if num cols exceeds max_cols

#create a APIConfig Key online free with quandl and use command -> quandl.ApiConfig.api_key = 'key_str'
quandl.ApiConfig.api_key = 'FU4P32QPSN87TtrNaxML'
df = quandl.get('WIKI/AMZN')
df = df[['Adj. Close']]

forecast_out = int(30)
df['Prediction'] = df[['Adj. Close']].shift(-forecast_out)

X = np.array(df.drop(['Prediction'],1))
X = preprocessing.scale(X) # used to standardize a dataset along any axis

X_forecast = X[-forecast_out:]
X = X[:-forecast_out]
y = np.array(df['Prediction'])
y = y[:-forecast_out]


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2)

clf = LinearRegression() # implement attributes of the LinearRegression object
clf.fit(X_train, y_train)  # standard fit function with the y_train sample weight being set to 1

confidence = clf.score(X_test, y_test)# score function takes into account natural accuracy as well as the time for the prediction to be given to
print('confidence: ', confidence)#prints the confidence score onto the terminal, using formula -> (correct predictions / total predictions) * 100%

forecast_prediction = clf.predict(X_forecast)
print(forecast_prediction)

#line graph prediction

days = [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30]
#new matplot addon allows for visualization of 30 day projected stock trend as a line graph
plt.plot(days,forecast_prediction,color='b')#input one is the x_axis, number two is the y_axis and the final one is the color
plt.xlabel('Days from Today')#label x axis as 'Days from Today'
plt.ylabel('Forecasted Stock Projection')#label y axis as 'Forecasted Stock Projection'
plt.show()#displays the visualization on the users device