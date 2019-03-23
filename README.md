Problem Statement
We will forecast the temperature of a particular location based on the time series analysis. The accuracy of the model must be greater than 95%.
Data Preprocessing
1.	Data Extraction: Pandas is a great resource for extracting and analyzing the data.
2.	Understanding the Data: Using head method we can see first few rows (by default 5) of the dataframe.
3.	Data Cleaning I decided to replace NaNs with the previous present data point. There are other methods to deal with NaNs such as removing rows with missing values, impute missing values, or applying machine learning algorithms such as K-Nearest neighbors.
Visualizing data:
I used the Matplotlib library to plot the data.

Indexing with Time Series:
 Data In time series analysis, time stamp act as index. Our current datetime data can be tricky to work with, therefore, we are using the start of each hour as the timestamp.
 
Checking Stationarity:
Stationarity is defined using very strict criterion. However, for practical purposes we can assume the series to be stationary if it has constant statistical properties over time, ie. the following:
1.	constant mean
2.	constant variance
3.	an autocovariance that does not depend on time.

We need to check the stationarity of data as this is the assumption of time series model.
There are two ways to check the stationarity:
1.	Rolling Statistics
2.	Dicky Fuller Test
If there exist stationarity in the data, then we can proceed by fitting the model or else if there exists some kind of trends and seasonality in the data, It must be removed then only we can fit the model.

Trends and Seasonality
Lets understand what is making a TS non-stationary. There are 2 major reasons behind non-stationaruty of a TS:
1. Trend – varying mean over time. For eg, in this case we saw that on average, the number of passengers was growing over time.
2. Seasonality – variations at specific time-frames. eg people might have a tendency to buy cars in a particular month because of pay increment or festivals.

For eliminating trends and seasonality we can do:
1.	Differencing: One of the most common methods of dealing with both trend and seasonality is differencing. In this technique, we take the difference of the observation at a particular instant with that at the previous instant. This mostly works well in improving stationarity. First order differencing can be done in Pandas
2.	Decomposition: In this approach, both trend and seasonality are modeled separately and the remaining part of the series is returned. Here the trend, seasonality are separated out from data and we can model the residuals. Next we will check stationarity of residuals:

Forecasting a Time Series
ARIMA stands for Auto-Regressive Integrated Moving Averages. The ARIMA forecasting for a stationary time series is nothing but a linear (like a linear regression) equation. The predictors depend on the parameters (p,d,q) of the ARIMA model:
1.	Number of AR (Auto-Regressive) terms (p): AR terms are just lags of dependent variable. 
2.	Number of MA (Moving Average) terms (q): MA terms are lagged forecast errors in prediction equation.
3.	Number of Differences (d): These are the number of nonseasonal differences, i.e. in this case we took the first order difference. So either we can pass that variable and put d=0 or pass the original variable and put d=1. Both will generate same results.


