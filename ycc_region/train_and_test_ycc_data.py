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
import codecs
import csv
import urllib2,json

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

import time
import sys
from twilio.rest import Client

#######################################################################################
######################## Messaging System #############################################

#ACCOUNT_SID ="AC66c09167d2d1d6d4a269c88c0f9f4ddf" # Put your Twilio account SID here
#AUTH_TOKEN ="e4b59d20658b12d940f920490b8834fd" # Put your auth token here
#twilioNumber = "+12015741913"

ACCOUNT_SID ="AC27d4f8da67dfcafbe085a674a981a499" # Toothless
AUTH_TOKEN ="bce1034827919c3e8d59b9a449a0de29" # Toothless
twilioNumber = "+17734677272" # Toothless

client = Client(ACCOUNT_SID, AUTH_TOKEN)

NUMBERS = ['+959950750660']

#,'+9595127360','+959970694000','+9595127360','+959443189540','+95973189540','+959254017119','+959977921242']
############################################################################################

#READ_API_KEY='5PXZFU85KTGOD1UB' # Lab_Two
#CHANNEL_ID='806195'   # Lab_Two

READ_API_KEY = 'NOZVI5WXI6HPLI9X' # Main Building
CHANNEL_ID = '804809'

def get_last_entry_id():
	conn = urllib2.urlopen("https://api.thingspeak.com/channels/804809/feeds.json?results=1")

	response = conn.read()

	data=json.loads(response)

	input_data = data['feeds']

	input_data = input_data[0]

	real_temperature = float(input_data['field1'])
	real_humidity = float(input_data['field2'])
	entry_id = input_data['entry_id']

	entry_id = int(entry_id)
	return entry_id,real_temperature,real_humidity

def series_to_supervised(data,n_in=1,n_out=1,dropnan=True):
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


##########################################################################################################################
############################################ Create Train and Testing Data From CSV File ##########################################
# load dataset
#dataset = read_csv('Total_year_data.csv',header=0,index_col=2)
#dataset = pd.read_csv('Total_year_data.csv',skipinitialspace=True,usecols=fields)
dataset = pd.read_csv('Datasets/ycc_weather.csv')

# Drop Columns
#dataset = dataset.drop(['created_at','entry_id','latitude','longitude','elevation','status'],axis=1)
dataset = dataset.drop(['Date','id','latitude','longitude','elevation','status'],axis=1)
dataset = dataset.dropna()
dataset = dataset.mask(dataset.eq('None')).dropna()

column_names = dataset.columns

temp_humid = dataset.values
print("===== Length of Data =====")
print(len(temp_humid))
print(temp_humid[0:3])
print(type(temp_humid))

##########################################################################################
######################### Data Preprocessing For Missing Values #############################

count = 0

one_sequence_np = []
total_weather = np.array([])

for j in range(0,len(temp_humid)):
	each_data = temp_humid[j]

	if j==0:
		for k in range(0,len(each_data)):

			temp_data = float(each_data[k])
			one_sequence_np = one_sequence_np + [temp_data]

		one_sequence_np = np.asarray(one_sequence_np)
		total_weather = one_sequence_np
		print("=== First ==")
		print(total_weather)
		one_sequence_np = []
	
	elif j== 1:
		for k in range(0,len(each_data)):
			temp_data = float(each_data[k])
			one_sequence_np = one_sequence_np + [temp_data]

		one_sequence_np = np.asarray(one_sequence_np)
		#print("---- Second Data -----")
		#print(one_sequence_np)
		total_weather = np.array((total_weather,one_sequence_np))
		one_sequence_np = []
		#print("---- Total after Second ----")
		#print(total_weather)
	else:
		for k in range(0,len(each_data)):
			temp_data = float(each_data[k])
			one_sequence_np = one_sequence_np + [temp_data]

		one_sequence_np = np.asarray(one_sequence_np)
		#print("---- Third Data -----")
		#print(one_sequence_np)
		total_weather = np.row_stack((total_weather,one_sequence_np))
		one_sequence_np = []
		#print("---- Last -----")
		#print(total_weather)
########################################################################################
#################################### Normalize Features ###########################################
temp_humid = total_weather.astype('float32')
print("=== Change Float Type ===")
print(temp_humid)

scaler = MinMaxScaler(feature_range=(0,1))
temp_humid_fit = scaler.fit(temp_humid)
print("=== Min values and Max values of MinMaxScaler ==== ")
print(scaler.data_max_)

temp_humid_encode = scaler.transform(temp_humid)
print("=== After Encode ===")
print(temp_humid_encode)

temp_humid_supervised = series_to_supervised(temp_humid_encode,1,1)

#split into train and test sets
supervised_values = temp_humid_supervised.values
print("=== supervised_values === ")
print(supervised_values)

n_train_hours = 2534

train = supervised_values[:n_train_hours,:]
test = supervised_values[n_train_hours:,:]

print("===== Train =====")
print(len(train))
print(train)
print("===== Test =====")
print(len(test))
print(test[0:5])

#split into input and outputs
train_X, train_y = train[:,:-2], train[:,-2:]
test_X , test_y  = test[:,:-2],test[:,-2:]

#print("=== Test X ===")
#print(test_X[0:5])
#print(type(test_X))

# reshape input to be 3D [samples, timesteps, features]
train_X = train_X.reshape((train_X.shape[0],1, train_X.shape[1]))
test_X = test_X.reshape((test_X.shape[0],1,test_X.shape[1]))

################################################# End of creating Training and Test Set ##########################################################
####################################################################################################################
'''
######################################################## Create MODEL #####################################################################
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
model.add(Dropout(0.4))
model.add(Dense(2))
model.compile(loss='mean_squared_error',optimizer='rmsprop',metrics=["accuracy"])
 
#model.summary()
 
# fit network
history = model.fit(train_X,train_y,epochs= 125,batch_size=64,validation_data=(test_X,test_y),verbose=2,shuffle=False)
 
model.save('Models/ycc_lstm_model_version_three.h5')

# summarize history for loss
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train','test'],loc='upper right')
plt.show()
'''
############################################ End of Creating Model ####################################################################
#####################################################################################################################
###################################### Model Load And Test with TEST Data #######################################################
'''
# load model from single file
model = load_model('Models/ycc_lstm_model_version_three.h5')

# make a prediction
yhat = model.predict(test_X)
predict_temp_humidity = scaler.inverse_transform(yhat)
predict_temp_humidity = predict_temp_humidity[:,:]

real_temp_humidity = scaler.inverse_transform(test_y)
real_temp_humidity = real_temp_humidity[:,:]

# calculate RMSE
rmse = np.sqrt(mean_squared_error(yhat,test_y))
print('Test RMSE: %.3f ' % rmse)

# Data Visualization Generalization

#------------------- Temperature ---------------------------------
aa = [x for x in range(50)]
plt.plot(aa, yhat[:50,0],marker='.',label="actual")
plt.plot(aa,test_y[:50,0],'r',marker='.',label="prediction")
plt.ylabel('Temperature',size=15)
plt.xlabel('Time step',size=15)
plt.legend(fontsize=15)
plt.show()


#------------------- Humidity ---------------------------------
aa = [x for x in range(50)]
plt.plot(aa, yhat[:50,1],marker='.',label="actual")
plt.plot(aa,test_y[:50,1],'r',marker='.',label="prediction")
plt.ylabel('Humidity',size=15)
plt.xlabel('Time step',size=15)
plt.legend(fontsize=15)
plt.show()
'''
'''
########################################## End of Test Data #################################################################
##################################################################################################################
########################## Load Model Accuracy ###########################################
model = load_model("Models/ycc_lstm_model_version_three.h5")

print(model.evaluate(train_X,train_y,verbose=0))
######################## End of  Model Accuracy #####################################
'''


#############################################################################################
########################### Test Real Time ############################################

# load model from single file
model = load_model('Models/ycc_lstm_model_version_three.h5')

conn = urllib2.urlopen("https://api.thingspeak.com/channels/804809/feeds.json?results=1")

response = conn.read()

data=json.loads(response)

input_data = data['feeds']

input_data = input_data[0]

last_temperature = input_data['field1']
last_humidity = input_data['field2']

last_entry_id = int(input_data['entry_id'])
last_temperature = float(last_temperature)
last_humidity = float(last_humidity)

#print("=== Income Entry ID ===")
#print(last_entry_id)
#print("=== Income Temperature ===")
#print(last_temperature)
#print("=== Income Humidity ===")
#print(last_humidity)

real_time_test_X = np.array([[last_temperature,last_humidity]])

real_time_test_X = real_time_test_X.astype('float32')
#print("=== Change to Float ===")
#print(real_time_test_X)

#print("=== Min values and Max values of MinMaxScaler ==== ")
#print(scaler.data_max_)
real_time_test = scaler.transform(real_time_test_X)
#print("=== Real Time Test after transform ===")
#print(real_time_test)

# Reshape Real Time Data
real_time_test = real_time_test.reshape((real_time_test.shape[0],1,real_time_test.shape[1]))
#print("=== After Reshape ===")
#print(real_time_test)

predict_temp_humidity = model.predict(real_time_test)
predict_temp_humidity = scaler.inverse_transform(predict_temp_humidity)
predict_temp_humidity = predict_temp_humidity[0]

predict_temperature = predict_temp_humidity[0]
predict_humidity = predict_temp_humidity[1]

real_data_entry_id = 0

while True:
	print("@@@@",last_entry_id)
	time.sleep(60)
	real_data_entry_id,real_data_temperature,real_data_humidity = get_last_entry_id()

	print("####Real ID ####")
	print(real_data_entry_id)
	print("**** Last ID **** ")
	print(last_entry_id)
	if real_data_entry_id>last_entry_id:
		break

print("=== Real Temperature and Humidity ===")
print("Real Time Entry ID " + str(real_data_entry_id))
print("Real Temperature " + str(real_data_temperature))
print("Real Humidity " + str(real_data_humidity))

print("=== Prediction Temperature and Humidity ==="	)
print("Predict after Entry ID " + str(last_entry_id))
print("Predict Temperature " + str(predict_temperature))
print("Predict Humidity " + str(predict_humidity))

rmse = np.sqrt(mean_squared_error([predict_temp_humidity],[[real_data_temperature,real_data_humidity]]))
print('Test RMSE: %.3f ' % rmse)

user_messages = "Temperature(actual,predict): ("+ str(real_data_temperature) + "," + str(predict_temperature) + "), Humidity(actual,predict): (" +  str(real_data_humidity) + "," + str(predict_humidity) + ")"  

for number in NUMBERS:
	message = client.messages.create(
	        to=number, 
            from_=twilioNumber, 
            body=user_messages)

'''
#####################################################################################
############################# Test Real Time with Function #############################
def test_real_time():
	print("Calling Real Time Method")
	# load model from single file
	model = load_model('ycc_lstm_model.h5')

	# YCC Lab Two Channel
	#conn = urllib2.urlopen("https://api.thingspeak.com/channels/806195/feeds.json?results=1")

	# YCC Main Building Channel
	conn = urllib2.urlopen("https://api.thingspeak.com/channels/804809/feeds.json?results=1")

	response = conn.read()

	data=json.loads(response)

	input_data = data['feeds']

	input_data = input_data[0]

	last_temperature = input_data['field1']
	last_humidity = input_data['field2']

	print("@@@@@@@",last_temperature,type(last_temperature))
	print("@@@@@@@",last_humidity,type(last_humidity))

	last_entry_id = int(input_data['entry_id'])
	last_temperature = float(last_temperature)
	last_humidity = float(last_humidity)

	print("=== Income Entry ID ===")
	print(last_entry_id)
	print("=== Income Temperature ===")
	print(last_temperature)
	print("=== Income Humidity ===")
	print(last_humidity)

	real_time_test_X = np.array([[last_temperature,last_humidity]])

	real_time_test_X = real_time_test_X.astype('float32')
	#print("=== Change to Float ===")
	#print(real_time_test_X)

	print("=== Min values and Max values of MinMaxScaler ==== ")
	print(scaler.data_max_)
	real_time_test = scaler.transform(real_time_test_X)
	print("=== Real Time Test after transform ===")
	print(real_time_test)

	# Reshape Real Time Data
	real_time_test = real_time_test.reshape((real_time_test.shape[0],1,real_time_test.shape[1]))
	print("=== After Reshape ===")
	print(real_time_test)

	predict_temp_humidity = model.predict(real_time_test)
	predict_temp_humidity = scaler.inverse_transform(predict_temp_humidity)
	predict_temp_humidity = predict_temp_humidity[0]

	predict_temperature = predict_temp_humidity[0]
	predict_humidity = predict_temp_humidity[1]

	real_data_entry_id = 0
	
	real_data_entry_id = 0
	
	while real_data_entry_id<last_entry_id:
		real_data_entry_id,real_data_temperature,real_data_humidity = get_last_entry_id()


	print("=== Real Temperature and Humidity ===")
	print("Real Time Entry ID " + str(real_data_entry_id))
	print("Real Temperature " + str(real_data_temperature))
	print("Real Humidity " + str(real_data_humidity))
	print("=== Prediction Temperature and Humidity ==="	)
	print("Predict after Time Entry ID " + str(last_entry_id))
	print("Predict Temperature " + str(predict_temperature))
	print("Predict Humidity " + str(predict_humidity))

	actual_temp_humidity = [real_data_temperature,real_data_humidity]

	rmse = np.sqrt(mean_squared_error(predict_temp_humidity,actual_temp_humidity))

	print("=== RETURN ===")
	print(predict_temp_humidity)
	print(actual_temp_humidity)
	print(rmse)

	return rmse,actual_temp_humidity,predict_temp_humidity
'''