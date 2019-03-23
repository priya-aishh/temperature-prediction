# -*- coding: utf-8 -*-
"""
Created on Mon Jan 28 22:41:23 2019

@author: priya
"""

import pandas as pd
import dateutil
import numpy as np
import matplotlib.pylab as plt
from datetime import datetime

from statsmodels.tsa.arima_model import ARIMA
from sklearn.metrics import mean_squared_error
from math import sqrt

print("pandas version:",pd.__version__)

#%matplotlib_inline
from matplotlib.pylab import rcParams
rcParams['figure.figsize'] = 12, 8

data = pd.read_csv("C:\\Users\\priya\\OneDrive\\Documents\\Task\\task-1.csv")
print(data)

data['Time'] = data['Time'].str[0:-23]

data = data.fillna(method='ffill')

data['Time'] = pd.to_datetime(data['Time'],infer_datetime_format=True)
data = data.set_index(data['Time'])

del data['Time']
print(data)

plt.xlabel("Time")
plt.ylabel("Temp.(Degree Celcius)")
plt.plot(data)
plt.show()

from statsmodels.tsa.stattools import adfuller

def test_stationarity(timeseries):
    #rolling statistics test
    rolmean = data.rolling(window=24).mean()
    rolstd = data.rolling(window=24).std()
    print(rolmean,rolstd)

    orig = plt.plot(data,color='blue',label='original')
    mean = plt.plot(rolmean,color='red',label='rolling mean')
    std = plt.plot(rolstd,color='black',label='rolling std dev')
    plt.legend(loc='best')
    plt.title("Rolling mean and std deviation")
    plt.show(block=False)

    #dickey-fuller test
    dftest = adfuller(data['Temp. (Degree Celcius)'], autolag='AIC')
    dfoutput = pd.Series(dftest[0:4], index=['Test Statistic','p-value','#Lags Used','Number of Observations Used'])
    for key,value in dftest[4].items():
        dfoutput['Critical Value (%s)'%key] = value
    print(dfoutput)

print("\n original data")
test_stationarity(data)

ts_log = np.log(data)
plt.plot(ts_log)

moving_avg = ts_log.rolling(window=24).mean()
plt.plot(ts_log)
plt.plot(moving_avg, color='red')

ts_log_moving_avg_diff = ts_log - moving_avg
ts_log_moving_avg_diff.head(24)

ts_log_moving_avg_diff.dropna(inplace=True)
print("\n after applying moving average")
test_stationarity(ts_log_moving_avg_diff)

expwighted_avg = ts_log.ewm(alpha=0.55,min_periods=24).mean()

plt.plot(ts_log)
plt.plot(expwighted_avg, color='red')

ts_log_ewma_diff = ts_log - expwighted_avg
print("after taking exponential weighted moving average")
test_stationarity(ts_log_ewma_diff)

ts_log_diff = ts_log - ts_log.shift()
plt.plot(ts_log_diff)

ts_log_diff.dropna(inplace=True)
print("\n after diffrenciation")
test_stationarity(ts_log_diff)

from statsmodels.tsa.seasonal import seasonal_decompose
decomposition = seasonal_decompose(ts_log,freq=24)

trend = decomposition.trend
seasonal = decomposition.seasonal
residual = decomposition.resid

print("\n after decomposition")
plt.subplot(411)
plt.plot(ts_log, label='Original')
plt.legend(loc='best')
plt.subplot(412)
plt.plot(trend, label='Trend')
plt.legend(loc='best')
plt.subplot(413)
plt.plot(seasonal,label='Seasonality')
plt.legend(loc='best')
plt.subplot(414)
plt.plot(residual, label='Residuals')
plt.legend(loc='best')
plt.tight_layout()

ts_log_decompose = residual
ts_log_decompose.dropna(inplace=True)
test_stationarity(ts_log_decompose)

print("In this case there is no visible trends and seasonality.")
print("Still we can compare the result of original and tranformed data")

#Forecasting the time series
#ACF and PACF plots:
from statsmodels.tsa.stattools import acf, pacf

lag_acf1 = acf(data, nlags=20)
lag_pacf1 = pacf(data, nlags=20, method='ols')
lag_acf2 = acf(ts_log_diff, nlags=20)
lag_pacf2 = pacf(ts_log_diff, nlags=20, method='ols')

print("For original data")
#Plot ACF: 
plt.subplot(121) 
plt.plot(lag_acf1)
plt.axhline(y=0,linestyle='--',color='gray')
plt.axhline(y=-1.96/np.sqrt(len(ts_log_diff)),linestyle='--',color='gray')
plt.axhline(y=1.96/np.sqrt(len(ts_log_diff)),linestyle='--',color='gray')
plt.title('Autocorrelation Function')
#Plot PACF:
plt.subplot(122)
plt.plot(lag_pacf1)
plt.axhline(y=0,linestyle='--',color='gray')
plt.axhline(y=-1.96/np.sqrt(len(ts_log_diff)),linestyle='--',color='gray')
plt.axhline(y=1.96/np.sqrt(len(ts_log_diff)),linestyle='--',color='gray')
plt.title('Partial Autocorrelation Function')
plt.tight_layout()
plt.show()

print("For transformed data")
#Plot ACF: 
plt.subplot(121) 
plt.plot(lag_acf2)
plt.axhline(y=0,linestyle='--',color='gray')
plt.axhline(y=-1.96/np.sqrt(len(ts_log_diff)),linestyle='--',color='gray')
plt.axhline(y=1.96/np.sqrt(len(ts_log_diff)),linestyle='--',color='gray')
plt.title('Autocorrelation Function')
#Plot PACF:
plt.subplot(122)
plt.plot(lag_pacf2)
plt.axhline(y=0,linestyle='--',color='gray')
plt.axhline(y=-1.96/np.sqrt(len(ts_log_diff)),linestyle='--',color='gray')
plt.axhline(y=1.96/np.sqrt(len(ts_log_diff)),linestyle='--',color='gray')
plt.title('Partial Autocorrelation Function')
plt.tight_layout()
plt.show()

# split into train and test sets
def training_model(part,no_month,dataset):
    X = dataset.values
    train, test = X[0:-part], X[-part:]
    history = [x for x in train]
    predictions1 = list()
    predictions2 = list()
    predictions3 = list()

    #AR MODEL
    for t in range(len(test)):
        model = ARIMA(history, order=(1,1,0))
        model_fit = model.fit(disp=-1)
        output = model_fit.forecast()
        pred1 = output[0]
        predictions1.append(pred1)
        obs = test[t]
        history.append(obs)
    #print('predicted=%f, expected=%f' % (pred, obs))
    mse1 = mean_squared_error(test, predictions1)
    rmse1 = sqrt(mse1)
    
    print('\nAR model')
    print('Mean Squared Error: %.3f' % mse1)
    print('Root Mean Squared Error: %.3f' % rmse1)
    errors = abs(predictions1 - test)
    print('Mean Absolute Error:', round(np.mean(errors), 2), 'degrees.')
    mape = np.mean(100 * (errors / test))
    accuracy = 100 - mape
    print('Accuracy:', round(accuracy, 2), '%.')


    # plot
    plt.plot(test, color='blue')
    plt.plot(predictions1, color='red')
    plt.title('Prediction with %4.f month data: %.4f' %(no_month, rmse1))
    plt.show()
    
    #MA MODEL
    for t in range(len(test)):
        model = ARIMA(history, order=(0,1,1))
        model_fit = model.fit(disp=-1)
        output = model_fit.forecast()
        pred2 = output[0]
        predictions2.append(pred2)
        obs = test[t]
        history.append(obs)
    #print('predicted=%f, expected=%f' % (pred, obs))
    mse2 = mean_squared_error(test, predictions2)
    rmse2 = sqrt(mse2)
    
    print('\nMA model')
    print('Mean Squared Error: %.3f' % mse2)
    print('Root Mean Squared Error: %.3f' % rmse2)
    errors = abs(predictions2 - test)
    print('Mean Absolute Error:', round(np.mean(errors), 2), 'degrees.')
    mape = np.mean(100 * (errors / test))
    accuracy = 100 - mape
    print('Accuracy:', round(accuracy, 2), '%.')


    # plot
    plt.plot(test, color='blue')
    plt.plot(predictions2, color='red')
    plt.title('Prediction with %4.f month data: %.4f' %(no_month, rmse2))
    plt.show()
    
    #ARIMA MODEL
    for t in range(len(test)):
        model = ARIMA(history, order=(1,1,1))
        model_fit = model.fit(disp=-1)
        output = model_fit.forecast()
        pred3 = output[0]
        predictions3.append(pred3)
        obs = test[t]
        history.append(obs)
    #print('predicted=%f, expected=%f' % (pred, obs))
    mse3 = mean_squared_error(test, predictions3)
    rmse3 = sqrt(mse3)
    
    print('\nARIMA model')
    print('Mean Squared Error: %.3f' % mse3)
    print('Root Mean Squared Error: %.3f' % rmse3)
    errors = abs(predictions3 - test)
    print('Mean Absolute Error:', round(np.mean(errors), 2), 'degrees.')
    mape = np.mean(100 * (errors / test))
    accuracy = 100 - mape
    print('Accuracy:', round(accuracy, 2), '%.')


    # plot
    plt.plot(test, color='blue')
    plt.plot(predictions3, color='red')
    plt.title('Prediction with %4.f month data: %.4f' %(no_month, rmse3))
    plt.show()

print("\n result of original data")
training_model(3312,3,data)

print("\n result of transformed data")
training_model(3312,3,ts_log_diff)
