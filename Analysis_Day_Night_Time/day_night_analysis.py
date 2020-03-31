import codecs
import csv
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt # this is used for the plot the graph 
import seaborn as sns # used for plot interactive graph. 


total_data_np = np.array([])
one_data_np = np.array([])

count  = 0
ten_count = 0
flag = True

ten_temperature_np = np.array([])
ten_humidity_np = np.array([])

temperature_np = np.array([])
humidity_np = np.array([])


with codecs.open('mainbuilding.csv','r') as f_reader:
	
	for line in f_reader:
		line = line.replace("\t"," ")
		data = line.split()

		if count == 0:
			for k in range(0,len(data)):
				if int(data[0])%10 == 0 and flag:
					flag = False
					ten_count = ten_count + 1
					ten_temperature_np = np.append(ten_temperature_np,float(data[1]))
					ten_humidity_np = np.append(ten_humidity_np,float(data[2]))

				if k!=0:
					# Temperature Data
					if k==1:
						temperature_np = np.append(temperature_np,float(data[k]))
					# Humidity Data
					elif k==2:
						humidity_np = np.append(humidity_np,float(data[k]))

					one_data_np = np.append(one_data_np,float(data[k]))
			
			count = count + 1

		elif count == 1:
			flag = True
			for k in range(0,len(data)):
				if int(data[0])%10 == 0 and flag:
					flag = False
					ten_count = ten_count + 1
					ten_temperature_np = np.append(ten_temperature_np,float(data[1]))
					ten_humidity_np = np.append(ten_humidity_np,float(data[2]))
			
				if k!=0:
					# Temperature Data
					if k==1:
						temperature_np = np.append(temperature_np,float(data[k]))
					# Humidity Data
					elif k==2:
						humidity_np = np.append(humidity_np,float(data[k]))

					total_data_np = np.append(total_data_np,float(data[k]))
			
			total_data_np = np.row_stack((one_data_np,total_data_np))
			one_data_np = []
			count = count + 1

		else:
			flag = True
			for k in range(0,len(data)):
				if int(data[0])%10 == 0 and flag:
					flag = False
					ten_count = ten_count + 1
					ten_temperature_np = np.append(ten_temperature_np,float(data[1]))
					ten_humidity_np = np.append(ten_humidity_np,float(data[2]))
			
				if k!= 0:
					# Temperature Data
					if k==1:
						temperature_np = np.append(temperature_np,float(data[k]))
					# Humidity Data
					elif k==2:
						humidity_np = np.append(humidity_np,float(data[k]))

					one_data_np = np.append(one_data_np,float(data[k]))

			total_data_np = np.row_stack((total_data_np,one_data_np))
			one_data_np = []
			count = count + 1
	
'''		
print("****************************************************************************")
print(temperature_np)
print(type(temperature_np))
print(len(temperature_np))

print("***************************************************************************")
print(humidity_np)
print(type(humidity_np))
print(len(humidity_np))

print("Line Number: ",ten_count)
print("*** Temperature ***")
print(ten_temperature_np)
print(len(ten_temperature_np))
print("@@@ Humidity @@@")
print(ten_humidity_np)
print(len(ten_humidity_np))
'''

#------------------- Temperature ---------------------------------
aa = [x for x in range(795)]
plt.plot(aa, temperature_np[:795],marker=',',label="actual")
plt.ylabel('Temperature',size=200)
plt.xlabel('Time step',size=200)
plt.legend(fontsize=10)
plt.show()

#------------------- RainFall ---------------------------------
aa = [x for x in range(795)]
plt.plot(aa, humidity_np[:795],marker=',',label="actual")
plt.ylabel('RainFall',size=200)
plt.xlabel('Time step',size=200)
plt.legend(fontsize=10)
plt.show()
