import sys 
import numpy as np # linear algebra
from scipy.stats import randint
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv), data manipulation as in SQL
import matplotlib.pyplot as plt # this is used for the plot the graph 
import seaborn as sns # used for plot interactive graph. 
from sklearn.model_selection import train_test_split # to split the data into two parts
from sklearn.model_selection import KFold # use for cross validation
from sklearn.preprocessing import StandardScaler # for normalization
from sklearn.preprocessing import MinMaxScaler
from sklearn.pipeline import Pipeline # pipeline making
from sklearn.model_selection import cross_val_score
from sklearn.feature_selection import SelectFromModel
from sklearn import metrics # for the check the error and accuracy of the model
from sklearn.metrics import mean_squared_error,r2_score
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import MinMaxScaler
from pandas import read_csv
import nexmo
from socket import socket, gethostbyname, AF_INET, SOCK_DGRAM
import time
import sys
from twilio.rest import Client

PORT_NUMBER = 6000
SIZE = 1024

hostName = gethostbyname( '172.20.10.4' )

mySocket = socket( AF_INET, SOCK_DGRAM )
mySocket.bind( (hostName, PORT_NUMBER) )


# for Deep-learing:
import keras
from keras.layers import Dense
from keras.models import Sequential
from keras.utils import to_categorical
from keras.optimizers import SGD 
from keras.callbacks import EarlyStopping
from keras.utils import np_utils
import itertools
from keras.layers import LSTM
from keras.layers.convolutional import Conv1D
from keras.layers.convolutional import MaxPooling1D
from keras.layers import Dropout
from keras.models import load_model

account_sid ="AC66c09167d2d1d6d4a269c88c0f9f4ddf" # Put your Twilio account SID here
auth_token ="e4b59d20658b12d940f920490b8834fd" # Put your auth token here
client = Client(account_sid, auth_token)

def series_to_supervised(data,n_in=1,n_out=6,dropnan=True):
	n_vars = 1 if type(data) is list else data.shape[1]

	dff = pd.DataFrame(data)
	cols = list()
	names = list()

	# input sequence ( t-n,....t-1)
	for i in range(n_in,0,-1):
		cols.append(dff.shift(i))
		names += [('var%d(t-%d)' % (j+1,i)) for j in range(n_vars)]

	# forecast sequence ( t, t+1,....t+n)
	for i in range(0,n_out):
		cols.append(dff.shift(-i))
		if i == 0:
			names += [('var%d(t)' % (j+1)) for j in range(n_vars)]
		else:
			names += [('var%d(t+%d)' % (j+1,i)) for j in range(n_vars)]

	# put it all together
	agg = pd.concat(cols,axis=1)
	agg.columns = names

	# drop rows with NaN values
	if dropnan:
		agg.dropna(inplace=True)
	return agg


#fields = ['Temp_Max','Temp_Min','Rain_Fall','Humidity','Wind Speed','Wind Direction']

#print('----	Running LSTM_dATA_pREPARATION ---- ')

# load dataset
#dataset = read_csv('Total_year_data.csv',header=0,index_col=2)
#dataset = pd.read_csv('Total_year_data.csv',skipinitialspace=True,usecols=fields)

dataset = pd.read_csv('Total_year_data.csv',sep=',',parse_dates={'Date_Time':['Date','Time']},infer_datetime_format=True,low_memory=False,na_values=['nan','?'],index_col='Date_Time')

dataset = dataset.drop(['Temp_Min'],axis=1)

#print('----- dataset values -----')
#print(dataset.head())
values = dataset.values
#print('----- values -----')
#print(values)
#print(type(values))

# integer encode direction
encoder = LabelEncoder()
#print('----- bEFORE eNCODING -----')
#print(values[:,4])
temp_wind_direction_data =	encoder.fit(values[:,4])
values[:,4] = 	encoder.transform(values[:,4])

print(values[:,4])

#print('----- aFTER eNCODING -----')
#print(values[:,4])

# ensure all data is float
values = values.astype('float32')

# normalize features
scaler = MinMaxScaler(feature_range=(0,1))

scaled = scaler.fit(values)
scaled = scaler.transform(values)


# frame as supervised learning
reframed = series_to_supervised(scaled,1,1)

#split into train and test sets
values = reframed.values

n_train_hours = 3286

train = values[:n_train_hours,:]

test = values[n_train_hours:,:]

#validation_set = values[n_train_hours:]

#split into input and outputs
train_X, train_y = train[:,:-5], train[:,-5:]
test_X , test_y  = test[:,:-5],test[:,-5:]

# reshape input to be 3D [samples, timesteps, features]
train_X = train_X.reshape((train_X.shape[0],1, train_X.shape[1]))
test_X = test_X.reshape((test_X.shape[0],1,test_X.shape[1]))

'''
# Model Architecture
# 1.) LSTM with 100 neurons in the first visible layer
# 2.) dropout 20%
# 3.) 1 neuron in the output layer for predicting Global_active_power
# 4.) The input shape will be 1 time step with 7 features.
# 5.) Use the Mean Absolute Error (MAE) loss function and the efficient Adam version of stochastic gradient descent.
# 6.) Model will be fit for 20 training epochs with a batch size of 70.

model = Sequential()

model.add(LSTM(100,activation='linear',input_shape=(train_X.shape[1],train_X.shape[2]),return_sequences=True))
model.add(LSTM(100))
model.add(Dropout(0.2))
model.add(Dense(5))
model.compile(loss='mean_squared_error',optimizer='rmsprop',metrics=["accuracy"])

#model.summary()

# fit network
history = model.fit(train_X,train_y,epochs=125,batch_size=56,validation_data=(test_X,test_y),verbose=2,shuffle=False)
model.save('lstm_model.h5')

# summarize history for loss
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train','test'],loc='upper right')
plt.show()

# load model from single file
model = load_model('lstm_model.h5')

# make a prediction
yhat = model.predict(test_X)
#print('----- pREDICTION -----')
#print(yhat)
#print('----- sHAPE of yhat -----')
#print(yhat.shape)
inv_yhat = scaler.inverse_transform(yhat)
inv_yhat = inv_yhat[:,:]

#print('--- load data one by one ---')
temp_data = inv_yhat[0]

data_rain_fall = temp_data[1]

if data_rain_fall<0:
	data_rain_fall = 0
	temp_data[1]  = 0

data_temp_max = temp_data[0]
data_rain_fall = data_rain_fall
data_humidity = temp_data[2]
data_wind_speed = temp_data[3]
data_wind_direction = temp_data[4]

inv_yhat[0] = temp_data
#print('----- aFTER iNVERT sCALING fOR fORECAST -----')
#print(inv_yhat)

#print(data_temp_max)
#print(data_rain_fall)
#print(data_humidity)
#print(data_wind_speed)
#print(data_wind_direction)

inv_y = scaler.inverse_transform(test_y)
inv_y = inv_y[:,:]
#print('----- aFTER iNVERT sCALING fOR aCTUAL -----')
#print(inv_y)

for j in inv_y:
	temp_lis = j

#print("###@@@",temp_lis)
# Create List To combine Integer and String Data
Weather_Data_lis = []
for i in range(0,len(temp_lis)-1):
	Weather_Data_lis.append(str(temp_lis[i]))

#print("After Combination :",Weather_Data_lis)

# Wind Direction Data
temp_data = int(temp_lis[4])
# Convert Wind Direction Integer to String
wind_direction_data = encoder.inverse_transform([temp_data])
# Concat Wind Direction String to Weather Data
Weather_Data_lis.append(str(wind_direction_data[0]))
#print("After Invert Scaling For Actual")
#print(temp_lis)
#print("Final Results:",Weather_Data_lis)


# ----- Send message with Nexmo -----
receiver = '095127360' # The number at which you want to send to
message = 'Temperature ='+ Weather_Data_lis[0]+', Rain fall ='+Weather_Data_lis[1]+', Humidity ='+Weather_Data_lis[2]+', Wind Speed ='+Weather_Data_lis[3]+', Wind Direction ='+Weather_Data_lis[4];

client = nexmo.Client(key='eaf31a5c', secret='J7IZMdhs5incuOSS')

client.send_message({
    'from': 'Weather Data',
    'to': '9595127360',
    'text': message,
})
'''

############################## REAL TIME #########################################

# Load Model
model = load_model('lstm_model.h5')



while True:
	(data,addr) = mySocket.recvfrom(SIZE)
	temp_weather_data = data.split()
	real_time_temperature = float(temp_weather_data[0])
	real_time_humidity = float(temp_weather_data[1])
	real_time_wind_speed = float(temp_weather_data[2])
	real_time_rainfall = float(temp_weather_data[3])
	real_time_wind_direction = temp_weather_data[4]
	
	total_numpy_arr = np.array([[real_time_temperature,real_time_humidity,real_time_wind_speed,real_time_rainfall,real_time_wind_direction]])
	
	total_numpy_arr[:,4]=encoder.transform(total_numpy_arr[:,4])
	total_numpy_arr = total_numpy_arr.astype('float32')

	scaled = scaler.transform(total_numpy_arr)

	real_time_test = scaled.reshape((scaled.shape[0],1,scaled.shape[1]))

	# PREDICT REAL TIME
	prediction_result = model.predict(real_time_test)
	
	real_prediction_result = scaler.inverse_transform(prediction_result)
	real_prediction_result[:,4] = int(round(real_prediction_result[:,4]))

	temp_lis = []

	for j in real_prediction_result:
		for i in j:
			temp_lis.append(i)

	temp_wind_direction_data =int(temp_lis[4])
	predict_wind_direction_data = encoder.inverse_transform([temp_wind_direction_data])
	#print("!!!",predict_wind_direction_data)

	if temp_lis[3] < 2.0:
		temp_lis[3] = 0.0

	# ----- Send message with Nexmo -----
	receiver = '09950750660' # The number at which you want to send to
	message = 'Temperature ='+ str(temp_lis[0])+', Rain fall ='+str(temp_lis[3])+', Humidity ='+str(temp_lis[2])+', Wind Speed ='+str(temp_lis[1])+', Wind Direction ='+predict_wind_direction_data
	message = message[0]
	
	'''
	print(message)
	print(type(message))
	client = nexmo.Client(key='eaf31a5c', secret='J7IZMdhs5incuOSS')

	client.send_message({
    'from': 'Weather Data',
    'to': '959950750660',
    'text': message,
	})
	time.sleep(2)
	'''

	message = client.api.account.messages.create(
	to="+959950750660", # Put your cellphone number here
	from_="+12015741913", # Put your Twilio number here
	body=message)

'''
# calculate RMSE
rmse = np.sqrt(mean_squared_error(inv_y,inv_yhat))
print('Test RMSE: %.2f ' % rmse)

# To improve the model, one has to adjust epochs and batch_size.

#------------------- Temp Max ---------------------------------
aa = [x for x in range(1)]
plt.plot(aa, inv_y[:1,0],marker='*',label="actual")
plt.plot(aa,inv_yhat[:1,0],'r',marker='*',label="prediction")
plt.ylabel('Temperature Max',size=15)
plt.xlabel('Time step',size=15)
plt.legend(fontsize=15)
plt.show()


#------------------- RainFall ---------------------------------
aa = [x for x in range(1)]
plt.plot(aa, inv_y[:1,1],marker='*',label="actual")
plt.plot(aa,inv_yhat[:1,1],'r',marker='*',label="prediction")
plt.ylabel('RainFall',size=15)
plt.xlabel('Time step',size=15)
plt.legend(fontsize=15)
plt.show()

#------------------- Humidity ---------------------------------
aa = [x for x in range(1)]
plt.plot(aa, inv_y[:1,2],marker='*',label="actual")
plt.plot(aa,inv_yhat[:1,2],'r',marker='*',label="prediction")
plt.ylabel('Humidity',size=15)
plt.xlabel('Time step',size=15)
plt.legend(fontsize=15)
plt.show()


#------------------- Wind Speed ---------------------------------
aa = [x for x in range(1)]
plt.plot(aa, inv_y[:1,3],marker='*',label="actual")
plt.plot(aa,inv_yhat[:1,3],'r',marker='*',label="prediction")
plt.ylabel('Wind Speed',size=15)
plt.xlabel('Time step',size=15)
plt.legend(fontsize=15)
plt.show()

#------------------- Wind Direction ---------------------------------
aa = [x for x in range(1)]
plt.plot(aa, inv_y[:1,4],marker='*',label="actual")
plt.plot(aa,inv_yhat[:1,4],'r',marker='*',label="prediction")
plt.ylabel('Wind Direction',size=15)
plt.xlabel('Time step',size=15)
plt.legend(fontsize=15)
plt.show()
'''