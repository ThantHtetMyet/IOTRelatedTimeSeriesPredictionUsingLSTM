#----------------------------------------  sERVER  --------------------------------------------
import time
import sys
from socket import socket, AF_INET, SOCK_DGRAM


SERVER_IP = '192.168.8.100'
#SERVER_IP = '172.20.10.4'
PORT_NUMBER = 6000


mySocket = socket(AF_INET, SOCK_DGRAM)
mySocket.connect((SERVER_IP,PORT_NUMBER)) 
#  ---------------------------------------------------------------------------------------------------------



#----------------------------------------  wIND sPEED --------------------------------------------
import SocketServer
import subprocess
import sys
import threading
from thread import start_new_thread
import time
import RPi.GPIO as GPIO
import math
import time
import serial
#  ---------------------------------------------------------------------------------------------------------

# ------------------------- tEMPERATURE --- hUMIDITY ----------------------------------------
import sys
from urllib import urlopen
from time import sleep
import Adafruit_DHT as dht
import RPi.GPIO as GPIO
# ------------------------------------------------------------------------------------------------------------

#------------------------- wIND sPEED ----------------------------------------------------------------
# RPi.GPIO Layout verwenden (wie Pin-Nummern)
GPIO.setmode(GPIO.BOARD)

# Pin 3 (GPIO 0) auf Input setzen
GPIO.setup(3, GPIO.IN, pull_up_down=GPIO.PUD_UP)

imp_per_sec = 0
actual_windspeed_msec = 0
events = []
# ----------------------------------------------------------------------------------------------------------------------


#-------------------------------- wIND dIRECTION and rAIN fALL --------------------------------------------
count = 0
arduino = serial.Serial("/dev/ttyACM0",9600,timeout=1)
data = arduino.readline()
#------------------------------------------------------------------------------------------------------------------------


#-------------------------------- tEMPERATURE --- hUMIDITY --------------------------------------------
DHT = 4
# Enter Your API key here
myAPI = '1HOXI98S1KZA6AKU' 
# URL where we will send the data, Don't change it
baseURL = 'https://api.thingspeak.com/update?api_key=%s' % myAPI
#  ------------------------------------------------------------------------------------------------------------------------

# ------------------------- tEMPERATURE --- hUMIDITY ------------------------------------------------------
def DHT22_data():
	# Reading from DHT22 and storing the temperature and humidity
	humi, temp = dht.read_retry(dht.DHT22, DHT) 
	return(str(humi),str(temp))
# ------------------------------------------------------------------------------------------------------------------------

#------------------------- wIND sPEED ------------------------------------------------------------------------------
def interrupt(val):
        global imp_per_sec
        imp_per_sec += 1

GPIO.add_event_detect(3, GPIO.RISING, callback = interrupt, bouncetime = 5)

def ws100_imp_to_mpersec(val):
        #y = 8E-09x5 - 2E-06x4 + 0,0002x3 - 0,0073x2 + 0,4503x + 0,11

        y = float("8e-9") * math.pow(val,5) - float("2e-6") * math.pow(val,4) + float("2e-4") * math.pow(val,3) - float("7.3e-3") * math.pow(val,2) + 0.4503 * val + 0.11
        if y < 0.2:
                y = 0
        return y

def threadeval():
        global imp_per_sec
        global actual_windspeed_msec
        while 1:
                time.sleep(5)
		# -------------- Wind Direction and RainFall -------------------
		data = arduino.readline()
   		pieces =data.split("\t")
  		dIRECTION = pieces[0]
		rAINFALL =    pieces[1]
		rain_fall = str(rAINFALL)
		#print("####",dIRECTION)
		if dIRECTION == "1":
			wd = "NE"
		elif dIRECTION =="2":
			wd = "E"
		elif dIRECTION =="3":
			wd = "SE"
		elif dIRECTION =="4":
			wd = "S"
		elif dIRECTION =="5":
			wd = "SW"
		elif dIRECTION =="6":
			wd ="W"
		elif dIRECTION =="7":
			wd = "NW"
		elif dIRECTION =="8":
			wd ="N"
		else:
			wd="Trace"
		
		wind_direction = str(dIRECTION)
    		# --------------------------------------------------------------------------
		# -------------------------  Wind Speed    -------------------------------------
                actual_windspeed_msec = ws100_imp_to_mpersec(imp_per_sec)
                #print("actual_windspeed_msec %f" % actual_windspeed_msec)
		#--------------------------------------------------------------------------------------		
		# -------------- Temperature and Humidity ---------------------
                humi, temp = DHT22_data()
                wind_speed = str(actual_windspeed_msec)
		# --------------------------------------------------------------------------
		
		Total_Weather_Data = temp + " " + humi + " " +  rain_fall + " " + str(wind_speed) + " " + wd
		
		
		# If Reading is valid
                print("Temperature " + str(temp) +".\n")
		print("Humidity " + str(humi)+".\n")
		print("Rain fall " + rain_fall+".\n")
                print("Wind Speed " + wind_speed+".\n")
		#print(wind_direction)
		print("Wind Direction " + wd +".\n")
                print("-------------------------------------------------------------------------------------")
		#print(Total_Weather_Data)
		
		
		# Sending the data to thingspeak
                conn = urlopen(baseURL + '&field1=%s&field2=%s&field3=%s&field4=%s&field5=%s' % (temp, humi,wind_speed,wind_direction,rain_fall))
		
		# Closing the connection
                conn.close()
                
		imp_per_sec = 0
                
		
		for x in events:
                    x.set()
		
		if humi!= None or temp!=None or humi!= "None" or temp!="None" or wd!= "None" or wd!="None":
			mySocket.send(Total_Weather_Data)
		
		
		time.sleep(1)
	
start_new_thread(threadeval, ())

HOST = ''
PORT = 2400

########################################################################################################################################################
'''  One instance per connection.
     Override handle(self) to customize action. '''

class TCPConnectionHandler(SocketServer.BaseRequestHandler):
    def handle(self):
        global actual_windspeed_msec
        self.event = threading.Event()
        events.append(self.event)
        while 1:
            self.event.wait()
            self.event.clear()
            try:
                self.request.sendall('{"windspeed": %f, "time": "%s"}' % (actual_windspeed_msec,time.strftime('%X %x %Z')))
            except:
                break
        events.remove(self.event)

########################################################################################################################################################

class Server(SocketServer.ThreadingMixIn, SocketServer.TCPServer):
    # Ctrl-C will cleanly kill all spawned threads
    daemon_threads = True
    # much faster rebinding
    allow_reuse_address = True

    def __init__(self, server_address, RequestHandlerClass):
        SocketServer.TCPServer.__init__(\
        self,\
        server_address,\
        RequestHandlerClass)

########################################################################################################################################################
# -----------------------------------------------------------------------------------------------------------------------------------------------------

if __name__ == "__main__":
    server = Server((HOST, PORT), TCPConnectionHandler)
    # terminate with Ctrl-C
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        sys.exit(0)
########################################################################################################################################################
