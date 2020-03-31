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
import datetime
import codecs

def Direction_Encoding(input_arr):
	final_lis = []
	for i in input_arr:
		final_lis.append(wind_dir_encoding(i))

	return final_lis

def Direction_Decoding(input_arr):
	final_lis = []
	for i in input_arr:
		final_lis.append(wind_dir_decoding(i))

	return final_lis

def wind_dir_encoding(dir_data):
	if dir_data=="NE":
		return 1
	if dir_data=="E":
		return 2
	if dir_data=="SE":
		return 3
	if dir_data=="S":
		return 4
	if dir_data=="SW":
		return 5
	if dir_data=="W":
		return 6
	if dir_data=="NW":
		return 7
	if dir_data=="N":
		return 8
	if dir_data=="Calm":
		return 9

def wind_dir_decoding(dir_data):
	dir_data = int(round(dir_data))
	if dir_data>9:
		dir_data = 9
	if dir_data==1:
		return "NE"
	if dir_data==2:
		return "E"
	if dir_data==3:
		return "SE"
	if dir_data==4:
		return "S"
	if dir_data==5:
		return "SW"
	if dir_data==6:
		return "W"
	if dir_data==7:
		return "NW"
	if dir_data==8:
		return "N"
	if dir_data==9:
		return "Calm"

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

def file_writer(sentence):
	with codecs.open("Datasets/2018_pyinoolwin.csv","a") as f_writer:
		f_writer.write(sentence)

temp_pd = pd.read_csv("Datasets/Total_Temperature.csv",header=None)
humidity_pd = pd.read_csv("Datasets/Total_Humidity.csv",header=None)
rainfall_pd = pd.read_csv("Datasets/Total_RainFall.csv",header=None)
wind_dir_pd = pd.read_csv("Datasets/Total_Wind_Direction.csv",header=None)
wind_speed_pd = pd.read_csv("Datasets/Total_Wind_Speed.csv",header=None)

weather_data = pd.concat([temp_pd, humidity_pd,rainfall_pd,wind_speed_pd,wind_dir_pd], ignore_index=True,axis=1)

total_weather_data = weather_data.values

# Encode Wind Direction Data
dir_data = total_weather_data[:,4]
dir_data = Direction_Encoding(dir_data)

temperature = total_weather_data[:,0]
humidity = total_weather_data[:,1]
rainfall = total_weather_data[:,2]
wind_speed = total_weather_data[:,3]
#wind_direction= np.array(dir_data)
wind_direction= total_weather_data[:,4]

total_weather_data = {'Temperature':temperature,'Humidity':humidity,'RainFall':rainfall,'Wind_Speed':wind_speed,'Wind_Direction':wind_direction}
#four_weather_data = {'Temperature':temperature,'Humidity':humidity,'RainFall':rainfall,'Wind_Speed':wind_speed}
five_weather_data = pd.DataFrame(total_weather_data,columns=["Temperature","Humidity","RainFall","Wind_Speed","Wind_Direction"])
#five_weather_data = five_weather_data.dropna()

#five_weather_data = five_weather_data.dropna()

five_weather_data_values = five_weather_data.values
#five_weather_data_values = np.concatenate((five_weather_data_values,five_weather_data_values[0:4]),axis=0)
print(five_weather_data_values)
print(len(five_weather_data_values))


date = datetime.datetime(2018,1,1)
#date_only = str(date_time).split()
#date_only = date_only[0]
time = ["9:30","12:30","18:30"]

counter_total = 0

for i in range(0,365):
	date_only = date
	print(date_only)
	date_only = str(date_only).split()
	date_only = date_only[0]
	print("###",date_only)
	for j in range(0,3):
		achain = time[j]
		one_time = five_weather_data_values[counter_total]
		tempe = one_time[0]
		humi  = one_time[1]
		rainf = one_time[2]
		wind_s = one_time[3]
		wind_d = one_time[4]
		#print("@@@@",achain)
		sentence = date_only + "," + achain + "," + str(tempe) + "," + str(humi) + "," + str(rainf) + "," + str(wind_s) + "," + str(wind_d) + "\n"
		file_writer(sentence)
		counter_total = counter_total + 1
	date += datetime.timedelta(days=1)
    

# ensure all data is float
all_weather_values = five_weather_data.values
all_weather_values = five_weather_data_values.astype('float32')

five_weather_data_values = np.concatenate((all_weather_values,all_weather_values),axis=0)
five_weather_data_values = np.concatenate((five_weather_data_values,all_weather_values),axis=0)

# normalize features
scaler = MinMaxScaler(feature_range=(0,1))
min_max_value = scaler.fit(five_weather_data_values)

five_weather_data_arr = scaler.transform(five_weather_data_values)

supervised_weather_data = series_to_supervised(five_weather_data_arr,1,6)

#split into train and test sets
values = supervised_weather_data.values

n_train_hours = 2184

train = values[:n_train_hours,:]

test = values[n_train_hours:,:]
#validation_set = values[n_train_hours:]

#split into input and outputs
train_X, train_y = train[:,:-5], train[:,-5:]
test_X , test_y  = test[:,:-5],test[:,-5:]

# reshape input to be 3D [samples, timesteps, features]
train_X = train_X.reshape((train_X.shape[0],1, train_X.shape[1]))
test_X = test_X.reshape((test_X.shape[0],1,test_X.shape[1]))

################################################# End of creating Training and Test Set ##########################################################
####################################################################################################################
######################################################## Create MODEL #####################################################################
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
model.compile(loss='mean_squared_error',optimizer='rmsprop',metrics=['accuracy'])
 
#model.summary()
 
# fit network
history = model.fit(train_X,train_y,epochs=132,batch_size=32,validation_data=(test_X,test_y),verbose=2,shuffle=False)
 
model.save('pyinoolwin_lstm_model.h5')

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
model = load_model('pyinoolwin_lstm_model.h5')

# make a prediction
yhat = model.predict(test_X)
print(len(yhat))
predict_weather_inverse = scaler.inverse_transform(yhat)
predict_weather = predict_weather_inverse[:,:]

dir_weather = []
dir_weather = Direction_Decoding(predict_weather[:,4])
print("#### Direction ####")
print(dir_weather)
print(len(dir_weather))
'''

####################################################################################################################
################################ Data Smoothing Generator #############################################################
'''
date = datetime.datetime(2016,1,1)
#date_only = str(date_time).split()
#date_only = date_only[0]
time = ["9:30","12:30","18:30"]

counter_total = 0

for i in range(0,365):
	date_only = date
	print(date_only)
	date_only = str(date_only).split()
	date_only = date_only[0]
	print("###",date_only)
	for j in range(0,3):
		achain = time[j]
		one_time = predict_weather[counter_total]
		tempe = round(one_time[0],1)
		humi  = round(one_time[1],1)
		rainf = round(one_time[2],1)
		wind_s = round(one_time[3],1)
		wind_d = dir_weather[counter_total]
		if rainf <0:
			rainf = 0
		if wind_s<0:
			wind_s = 0
		#print("@@@@",achain)
		sentence = date_only + "," + achain + "," + str(tempe) + "," + str(humi) + "," + str(rainf) + "," + str(wind_s) + "," + str(wind_d) + "\n"
		file_writer(sentence)
		counter_total = counter_total + 1
	date += datetime.timedelta(days=1)
'''