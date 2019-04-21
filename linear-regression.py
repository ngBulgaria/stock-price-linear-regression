#first import all needed libraries
import numpy as np
import matplotlib.pyplot as plot
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from pandas_datareader import data
import datetime

#set stock
ticker = 'TSLA'
#set start and end date
start_date = datetime.date(2018,1,1)
end_date = datetime.date(2019,4,21)

#create chart name
plottitle1 = 'From '
plottitle2 = 'To '
finaltitle = ticker + ' ' + plottitle1 + start_date.strftime('%m/%d/%Y') + ' ' + plottitle2 + end_date.strftime('%m/%d/%Y')

#get data from Yahoo finance
df = data.get_data_yahoo(ticker, start_date, end_date)
df.head()

#prepare x and y axes
df = df.reset_index()
price = df['Adj Close'].tolist()
date = df.index.tolist()
 
#Vector
date = np.reshape(date, (len(date), 1))
price = np.reshape(price, (len(price), 1))

#LR Object
lr = LinearRegression()
lr.fit(date, price)

# Plot the prepared model
plot.plot(date, price, color='blue', label= 'Close Price') #plotting the initial datapoints
plot.plot(date, lr.predict(date), color='red', linewidth=2, label = 'Linear Regression') #plotting the line made by linear regression
plot.title(finaltitle)
plot.legend()
plot.xlabel('Day count')
plot.show()