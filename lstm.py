# Importing the libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


#import full dataset
#importing dataset
df = pd.read_csv('AMZN.csv')
data_set = df.iloc[:, 1:2].values

# Feature Scaling
from sklearn.preprocessing import MinMaxScaler
sc = MinMaxScaler(feature_range = (0, 1))
data_set_scaled = sc.fit_transform(data_set)
#print(len(data_set_scaled))



#splitting train and test set
train = 0.6
sp = int(train * len(data_set_scaled))
stock_train =data_set_scaled [0:sp]
stock_test=data_set_scaled [sp:]


# Creating a data structure with 60 timesteps and 1 output
timestep=50
X_train = []
y_train = []
for i in range(timestep, len(stock_train)):
    X_train.append(stock_train[i-timestep:i, 0])
    y_train.append(stock_train[i, 0])
X_train, y_train = np.array(X_train), np.array(y_train)


# Reshaping
X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))
# Part 2 - Building the RNN

# Importing the Keras libraries and packages
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import Dropout


regressor = Sequential()
# Adding the first LSTM layer and some Dropout regularisation
regressor.add(LSTM(units = 50, return_sequences = True, input_shape = (X_train.shape[1], 1)))
regressor.add(Dropout(0.2))

# Adding a second LSTM layer and some Dropout regularisation
regressor.add(LSTM(units = 50, return_sequences = True))
regressor.add(Dropout(0.2))

# Adding a third LSTM layer and some Dropout regularisation
regressor.add(LSTM(units = 50, return_sequences = True))
regressor.add(Dropout(0.2))

# Adding a fourth LSTM layer and some Dropout regularisation
regressor.add(LSTM(units = 50))
regressor.add(Dropout(0.2))

# Adding the output layer
regressor.add(Dense(units = 1))

# Compiling the RNN
regressor.compile(optimizer = 'adam', loss = 'mean_squared_error', metrics=['mean_absolute_error'])

# Fitting the RNN to the Training set
regressor.fit(X_train, y_train, epochs = 100, batch_size=32)


# Part 3 - Making the predictions and visualising the results

#for test data
inputs = data_set_scaled[len(data_set_scaled) - len(stock_test) - timestep:]
#inputs = inputs.reshape(-1,1)
#
#print(type(stock_test))
#print(type(input))

#
X_test = []
for i in range(timestep, timestep+len(stock_test)):
    X_test.append(inputs[i-timestep:i, 0])
X_test = np.array(X_test)
X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))



predicted_stock_price = regressor.predict(X_test)
predicted_stock_price = sc.inverse_transform(predicted_stock_price)


# Visualising the results
real_stock_price=sc.inverse_transform(stock_test)
plt.plot(real_stock_price, color = 'red', label = 'Real Amazon Stock Price')
plt.plot(predicted_stock_price, color = 'blue', label = 'Predicted Amazon Stock Price')
plt.title('Amazon Stock Price Prediction')
plt.xlabel('Time')
plt.ylabel('Amazon Stock Price')
plt.legend()
plt.show()


















































