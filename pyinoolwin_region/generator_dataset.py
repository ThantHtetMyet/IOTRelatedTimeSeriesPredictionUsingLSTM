import codecs
import pandas as pd
import math

humi_nine = pd.read_csv("Datasets/PYN_(_Humidity_9_30_).csv",header=None)
humi_twelve = pd.read_csv("Datasets/PYN_(_Humidity_12_30_).csv",header=None)
humi_six = pd.read_csv("Datasets/PYN_(_Humidity_18_30_).csv",header=None)

wind_dir_nine = pd.read_csv("Datasets/PYN_(_WIND_DIRECTION_9_30_AM_).csv",header=None)
wind_dir_twelve = pd.read_csv("Datasets/PYN_(_WIND_DIRECTION_12_30_AM_).csv",header=None)
wind_dir_six = pd.read_csv("Datasets/PYN_(_WIND_DIRECTION_18_30_AM_).csv",header=None)

wind_speed_nine = pd.read_csv("Datasets/PYN_(_WIND_SPEED_9_30_AM_).csv",header=None)
wind_speed_twelve = pd.read_csv("Datasets/PYN_(_WIND_SPEED_12_30_AM_).csv",header=None)
wind_speed_six = pd.read_csv("Datasets/PYN_(_WIND_SPEED_18_30_AM_).csv",header=None)

rainfall = pd.read_csv("Datasets/PYN_(_RAINFALL_).csv",header=None)

temperature = pd.read_csv("Datasets/PYN_(_TEMPERATURE_).csv",header=None)

humi_twelve_np = humi_twelve.values
humi_nine_np = humi_nine.values
humi_six_np = humi_six.values

wind_dir_nine_np = wind_dir_nine.values
wind_dir_twelve_np = wind_dir_twelve.values
wind_dir_six_np = wind_dir_six.values

wind_speed_nine_np = wind_speed_nine.values
wind_speed_twelve_np = wind_speed_twelve.values
wind_speed_six_np = wind_speed_six.values

rainfall_np = rainfall.values
temperature_np = temperature.values

time_lis = ["9:30","12:30","18:30"]
'''
with codecs.open("Datasets/Total_Wind_Speed.csv","w") as f_writer:
	for month in range(0,12):
		temp_data_nine = wind_speed_nine[month]
		temp_data_twelve = wind_speed_twelve[month]
		temp_data_six = wind_speed_six[month]

		for i in range(0,len(temp_data_nine)):
			data_nine = temp_data_nine[i]
			data_twelve = temp_data_twelve[i]
			data_six = temp_data_six[i]

			f_writer.write(str(data_nine) + "\n")
			f_writer.write(str(data_twelve) + "\n")
			f_writer.write(str(data_six) + "\n")

'''
'''
with codecs.open("Datasets/Total_RainFall.csv","w") as f_writer:
	for month in range(0,12):
		temp_data = rainfall[month]
		
		for i in range(0,len(temp_data)):
			data = temp_data[i]

			for k in range(0,3):
				f_writer.write(str(data) + "\n")

'''

temp_pd = pd.read_csv("Datasets/Total_Temperature.csv",header=None)
humidity_pd = pd.read_csv("Datasets/Total_Humidity.csv",header=None)
rainfall_pd = pd.read_csv("Datasets/Total_RainFall.csv",header=None)
wind_dir_pd = pd.read_csv("Datasets/Total_Wind_Direction.csv",header=None)
wind_speed_pd = pd.read_csv("Datasets/Total_Wind_Speed.csv",header=None)

print(temp_pd)
print(humidity_pd)

temp_np = temp_pd.values
rainfall_np = rainfall_pd.values
humidity_np = humidity_pd.values
wind_dir_np = wind_dir_pd.values
wind_speed_np = wind_speed_pd.values


#print(temp_np)
#print(rainfall_np)
#print(humidity_np)
#print(wind_dir_np)
#print(wind_speed_np.shape)
#total_data = {"Temperature":temp_np,"Humidity":humidity_np,"RainFall":rainfall_np,"WindSpeed":wind_speed_np,"WindDirection":wind_dir_np}
#total_data = [temp_pd,humidity_pd]
#all_data = pd.concat(temp_pd,humidity_pd)
#bigdata = temp_pd.append(humidity_pd, ignore_index=True)
bigdata = pd.concat([temp_pd, humidity_pd,rainfall_pd,wind_speed_pd,wind_dir_pd], ignore_index=True,axis=1)
print(bigdata)