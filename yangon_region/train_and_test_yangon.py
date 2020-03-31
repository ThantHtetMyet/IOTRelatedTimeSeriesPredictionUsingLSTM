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

#ACCOUNT_SID ="AC66c09167d2d1d6d4a269c88c0f9f4ddf" # Put your Twilio account SID here
#AUTH_TOKEN ="e4b59d20658b12d940f920490b8834fd" # Put your auth token here
#twilioNumber = "+12015741913"

ACCOUNT_SID ="AC27d4f8da67dfcafbe085a674a981a499" # Toothless
AUTH_TOKEN ="bce1034827919c3e8d59b9a449a0de29" # Toothless
twilioNumber = "+17734677272" # Toothless

client = Client(ACCOUNT_SID, AUTH_TOKEN)

NUMBERS = ['+959950750660']

#,'+9595127360','+959970694000','+9595127360','+959443189540','+95973189540','+959254017119','+959977921242']

##############################################################################
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

def sendMessage(text):
    #ACCOUNT_SID ="AC66c09167d2d1d6d4a269c88c0f9f4ddf" # Thant Htet Myet
    #AUTH_TOKEN ="e4b59d20658b12d940f920490b8834fd" # Thant Htet Myet
    #twilioNumber = "+12015741913" # Thant Htet Myet
    ACCOUNT_SID ="AC27d4f8da67dfcafbe085a674a981a499" # Toothless
    AUTH_TOKEN ="bce1034827919c3e8d59b9a449a0de29" # Toothless
    twilioNumber = "+17734677272" # Toothless
    
    client = Client(ACCOUNT_SID, AUTH_TOKEN)

    NUMBERS = ['+959950750660','+9595127360','+959970694000','+9595127360','+959443189540','+95973189540','+959254017119','+959977921242']

    for number in NUMBERS:
        message = client.messages.create(
            to=number, 
            from_=twilioNumber, 
            body=text)
        print message.sid

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


##########################################################################################################################
############################################ Create Train and Testing Data ##########################################

# load dataset
#dataset = read_csv('Total_year_data.csv',header=0,index_col=2)
#dataset = pd.read_csv('Total_year_data.csv',skipinitialspace=True,usecols=fields)

dataset = pd.read_csv('Datasets/yangon_weather.csv',sep=',',parse_dates={'Date_Time':['Date','Time']},infer_datetime_format=True,low_memory=False,na_values=['nan','?'],index_col='Date_Time')

dataset = dataset.drop(['Temp_Min'],axis=1)

#print('----- dataset values -----')
#print(dataset.head())
values = dataset.values

dir_data = Direction_Encoding(values[:,4])

temperature = values[:,0]
rainfall = values[:,1]
humidity = values[:,2]
wind_speed = values[:,3]
wind_direction= np.array(dir_data)

# Create Wind Direction Data Frame
#wind_direction_df = pd.DataFrame({'Wind_Direction':wind_direction})
#wind_direction_df = wind_direction_df.dropna()

total_weather_data = {'Temperature':temperature,'Humidity':humidity,'RainFall':rainfall,'Wind_Speed':wind_speed,'Wind_Direction':wind_direction}
#four_weather_data = {'Temperature':temperature,'Humidity':humidity,'RainFall':rainfall,'Wind_Speed':wind_speed}
five_weather_data = pd.DataFrame(total_weather_data,columns=["Temperature","Humidity","RainFall","Wind_Speed","Wind_Direction"])
five_weather_data = five_weather_data.dropna()

# Get Wind Direction before dropping
#wind_direction_np = five_weather_data['Wind_Direction'].values

#Drop Wind Direction Columns
#four_weather_data = five_weather_data.drop(columns=['Wind_Direction'])

#four_weather_data_values = four_weather_data.values
five_weather_data_values = five_weather_data.values

# ensure all data is float
#four_weather_data_values = four_weather_data_values.astype('float32')
five_weather_data_values = five_weather_data_values.astype('float32')

# normalize features
scaler = MinMaxScaler(feature_range=(0,1))
five_weather_data_arr = scaler.fit(five_weather_data_values)
five_weather_data_arr = scaler.transform(five_weather_data_values)

#four_weather_data_arr = scaler.fit(four_weather_data_values)
#four_weather_data_arr = scaler.transform(four_weather_data_values)

#five_weather_data = np.column_stack((four_weather_data_arr,wind_direction_np))
#print(five_weather_data)

reframed = series_to_supervised(five_weather_data_arr,1,5)
#split into train and test sets
values = reframed.values

n_train_hours = 3000

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
history = model.fit(train_X,train_y,epochs=125,batch_size=56,validation_data=(test_X,test_y),verbose=2,shuffle=False)
 
model.save('Models/yangon_lstm_model_version_three.h5')

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

# load model from single file
model = load_model('Models/yangon_lstm_model_version_three.h5')

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
plt.ylabel('Temperature Max',size=15)
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
'''
########################## Load Model Accuracy ###########################################
model = load_model("yangon_lstm_model.h5")

print(model.evaluate(train_X,train_y,verbose=0))
######################## End of  Model Accuracy #####################################
'''