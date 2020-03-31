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

######################### Read Raw Data ###################################
#temp_pd = pd.read_csv("Datasets/Total_Temperature.csv",header=None)
#humidity_pd = pd.read_csv("Datasets/Total_Humidity.csv",header=None)
#rainfall_pd = pd.read_csv("Datasets/Total_RainFall.csv",header=None)
#wind_dir_pd = pd.read_csv("Datasets/Total_Wind_Direction.csv",header=None)
#wind_speed_pd = pd.read_csv("Datasets/Total_Wind_Speed.csv",header=None)

#weather_data = pd.concat([temp_pd, humidity_pd,rainfall_pd,wind_speed_pd,wind_dir_pd], ignore_index=True,axis=1)
weather_data = pd.read_csv("Datasets/pyinoolwin_weather.csv")
weather_data = weather_data.drop(['Date','Time'],axis=1)

total_weather_data = weather_data.values

# Encode Wind Direction Data
dir_data = total_weather_data[:,4]
dir_data = Direction_Encoding(dir_data)

temperature = total_weather_data[:,0]
humidity = total_weather_data[:,1]
rainfall = total_weather_data[:,2]
wind_speed = total_weather_data[:,3]
wind_direction= np.array(dir_data)

total_weather_data = {'Temperature':temperature,'Humidity':humidity,'RainFall':rainfall,'Wind_Speed':wind_speed,'Wind_Direction':wind_direction}
#four_weather_data = {'Temperature':temperature,'Humidity':humidity,'RainFall':rainfall,'Wind_Speed':wind_speed}
five_weather_data = pd.DataFrame(total_weather_data,columns=["Temperature","Humidity","RainFall","Wind_Speed","Wind_Direction"])

five_weather_data = five_weather_data.dropna()

five_weather_data_values = five_weather_data.values
five_weather_data_values = np.concatenate((five_weather_data_values,five_weather_data_values[0:4]),axis=0)

# ensure all data is float
five_weather_data_values = five_weather_data.values
five_weather_data_values = five_weather_data_values[0:1095]
five_weather_data_values = five_weather_data_values.astype('float32')

#five_weather_data_values = np.concatenate((all_weather_values,all_weather_values),axis=0)
#five_weather_data_values = np.concatenate((five_weather_data_values,all_weather_values),axis=0)
#print(five_weather_data_values)
#print(type(five_weather_data_values))
#print(len(five_weather_data_values))

# normalize features
scaler = MinMaxScaler(feature_range=(0,1))
min_max_value = scaler.fit(five_weather_data_values)

five_weather_data_arr = scaler.transform(five_weather_data_values)

supervised_weather_data = series_to_supervised(five_weather_data_arr,1,6)

#split into train and test sets
values = supervised_weather_data.values

data_count = 1040
train = values[:data_count,:]
test = values[data_count:,:]

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
history = model.fit(train_X,train_y,epochs=125,batch_size=64,validation_data=(test_X,test_y),verbose=2,shuffle=False)
 
model.save('Models/pyinoolwin_lstm_model_version_three.h5')

# summarize history for loss
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train','test'],loc='upper right')
plt.show()
############################################ End of Creating Model ####################################################################
#####################################################################################################################
'''
###################################### Model Load And Test with TEST Data #######################################################
# load model from single file
model = load_model('Models/pyinoolwin_lstm_model_version_three.h5')

# make a prediction
yhat = model.predict(test_X)

predict_weather_inverse = scaler.inverse_transform(yhat)
predict_weather = predict_weather_inverse[:,:]

real_weather = scaler.inverse_transform(test_y)
real_weather = real_weather[:,:]

# calculate RMSE
rmse = np.sqrt(mean_squared_error(predict_weather,real_weather))
print('Test RMSE: %.3f ' % rmse)

#------------------- Temperature ---------------------------------
aa = [x for x in range(20)]
plt.plot(aa, real_weather[:20,0],marker='*',label="actual")
plt.plot(aa,predict_weather[:20,0],'r',marker='*',label="prediction")
plt.ylabel('Temperature',size=15)
plt.xlabel('Time step',size=15)
plt.legend(fontsize=15)
plt.show()


#------------------- RainFall ---------------------------------
aa = [x for x in range(20)]
plt.plot(aa, real_weather[:20,1],marker='*',label="actual")
plt.plot(aa,predict_weather[:20,1],'r',marker='*',label="prediction")
plt.ylabel('RainFall',size=15)
plt.xlabel('Time step',size=15)
plt.legend(fontsize=15)
plt.show()

#------------------- Humidity ---------------------------------
aa = [x for x in range(20)]
plt.plot(aa, real_weather[:20,2],marker='*',label="actual")
plt.plot(aa,predict_weather[:20,2],'r',marker='*',label="prediction")
plt.ylabel('Humidity',size=15)
plt.xlabel('Time step',size=15)
plt.legend(fontsize=15)
plt.show()


#------------------- Wind Speed ---------------------------------
aa = [x for x in range(20)]
plt.plot(aa, real_weather[:20,3],marker='*',label="actual")
plt.plot(aa,predict_weather[:20,3],'r',marker='*',label="prediction")
plt.ylabel('Wind Speed',size=15)
plt.xlabel('Time step',size=15)
plt.legend(fontsize=15)
plt.show()

#------------------- Wind Direction ---------------------------------
aa = [x for x in range(20)]
plt.plot(aa, real_weather[:20,4],marker='*',label="actual")
plt.plot(aa,predict_weather[:20,4],'r',marker='*',label="prediction")
plt.ylabel('Wind Direction',size=15)
plt.xlabel('Time step',size=15)
plt.legend(fontsize=15)
plt.show()
########################################## End of Test Data #################################################################
##################################################################################################################
