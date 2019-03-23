We will forecast the temperature of a particular location based on the time series analysis. The accuracy of the model must be greater than 95%. 

Data Preprocessing

1. Data Extraction: Pandas is a great resource for extracting and analyzing the data.

2. Understanding the Data:
Using head method we can see first few rows (by default 5) of the dataframe.

3. Data Cleaning
I decided to replace NaNs with the previous present data point. There are other methods to deal with NaNs such as removing rows with missing values, impute missing values, or applying machine learning algorithms such as K-Nearest neighbors.

Visulaizing data
I used the Matplotlib library to plot the data.

Indexing with Time Series Data
In time series analysis, time stamp act as index. Our current datetime data can be tricky to work with, therefore, we are using the start of each hour as the timestamp.

In time series analysis, the timestamp acts as a index of each row, for that we need to take timestamp as index and drop the datetime column from the data.
We need to check the stationarity of data as this is the assumption of time series model. There are two ways to check the stationarity:
1. Rolling Statistics
2. Dicky Fuller Test

If there exist stationarity in the data, then we can proceed by fitting the model or else if there exists some kind of trends and seasonality in the data, It must be removed then only we can fit the model.

For removing trends amd seasonality we can do:
1. Differentiation
2. Decomposition

In last step, Time series forecasting with AR, MA and ARIMA, And lastly we will compare the accuracy of each model.
