import pandas as pd
import dateutil
import numpy as np
import matplotlib.pylab as plt
from datetime import datetime

from statsmodels.tsa.arima_model import ARIMA
from sklearn.metrics import mean_squared_error
from math import sqrt

#%matplotlib_inline
from matplotlib.pylab import rcParams
rcParams['figure.figsize'] = 12, 8

data = pd.read_csv("temperature-prediction/data.csv")
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

ts_log = np.log(data)

ts_log_diff = ts_log - ts_log.shift()
ts_log_diff.dropna(inplace=True)

#Forecasting the time series
#ACF and PACF plots:
from statsmodels.tsa.stattools import acf, pacf
lag_acf = acf(ts_log_diff, nlags=20)
lag_pacf = pacf(ts_log_diff, nlags=20, method='ols')

#Plot ACF: 
plt.subplot(121) 
plt.plot(lag_acf)
plt.axhline(y=0,linestyle='--',color='gray')
plt.axhline(y=-1.96/np.sqrt(len(ts_log_diff)),linestyle='--',color='gray')
plt.axhline(y=1.96/np.sqrt(len(ts_log_diff)),linestyle='--',color='gray')
plt.title('Autocorrelation Function')

#Plot PACF:
plt.subplot(122)
plt.plot(lag_pacf)
plt.axhline(y=0,linestyle='--',color='gray')
plt.axhline(y=-1.96/np.sqrt(len(ts_log_diff)),linestyle='--',color='gray')
plt.axhline(y=1.96/np.sqrt(len(ts_log_diff)),linestyle='--',color='gray')
plt.title('Partial Autocorrelation Function')
plt.tight_layout()
plt.show()


# split into train and test sets
def training_model(part,no_month):
    X = data.values
    train, test = X[0:-part], X[-part:]
    history = [x for x in train]
    predictions = list()

    #ARIMA MODEL
    for t in range(len(test)):
        model = ARIMA(history, order=(1,1,0))
        model_fit = model.fit(disp=0)
        output = model_fit.forecast()
        pred = output[0]
        predictions.append(pred)
        obs = test[t]
        history.append(obs)
    #print('predicted=%f, expected=%f' % (pred, obs))
    mse = mean_squared_error(test, predictions)
    rmse = sqrt(mse)

    print('Mean Squared Error: %.3f' % mse)
    print('Root Mean Squared Error: %.3f' % rmse)

    # plot
    plt.plot(test, color='blue')
    plt.plot(predictions, color='red')
    plt.title('Prediction with %4.f month data: %.4f' %(no_month, rmse))
    plt.show()

training_model(3332,2)
