from Tkinter import *
from PIL import ImageTk,Image
from train_and_test_ycc_data import *
import tkMessageBox

radio_region = 0

def ycc_real_time_test():
	
	print("REAL TIME TESTING......")
	rmse,actual,predict = test_real_time()
	print(rmse,actual,predict)
	root_mean_square_error = str(rmse)
	
	actual_temp_data = str(actual[0])
	actual_humid_data = str(actual[1])
	actual_data_str = "Temperature:" + actual_temp_data + "," + "Humidity:" + actual_humid_data
	
	predict_temp_data = str(predict[0])
	predict_humid_data = str(predict[1])
	
	predict_data_str = "Temperature:" + predict_temp_data + "," + "Humidity:" + predict_humid_data
	
	total_msg = "RMSE:" + root_mean_square_error + "\n" + "PREDICTION:" + "\n" + predict_data_str + "\n" + "ACTUAL:" + "\n" + actual_data_str + "\n"

	result_frame = Frame(root,width=700,height=500)
	result_frame.grid(row=1,column=0)
	#button_frame.place(x=200,y=250,anchor="center")
	#img = ImageTk.PhotoImage(Image.open("/home/dell/Documents/wORKSPACE/GUI_tkinter/rmse_vs_batch.png"))
	Label(result_frame,text="------- Prediction Result -------").grid(row=0,column=0,sticky=N)	
	Label(result_frame,text=total_msg,fg="blue",font=("Helvetica", 22)).grid(row=1,column=0,sticky=S,padx=20,pady=20)
	

def ygn_next_page():
	for widget in ygn_frame.winfo_children():
		widget.destroy()

	ygn_next_frame = Frame(root)
	ygn_next_frame.grid(row=1,column=0,sticky="S")
	#button_frame.place(x=200,y=250,anchor="center")
	#img = ImageTk.PhotoImage(Image.open("/home/dell/Documents/wORKSPACE/GUI_tkinter/rmse_vs_batch.png"))
	
	Label(ygn_next_frame,text="Yangon region:",fg="blue",font=("Helvetica", 22)).grid(row=0,column=0,sticky=W,padx=20,pady=20)
	
	Button(ygn_next_frame,text="Back",command=ygn_back_page).grid(row=2,column=1,sticky="E",padx=5,pady=15)
	Button(ygn_next_frame,text="Home",command=home_page).grid(row=2,column=2,sticky="W",padx=2,pady=15)
	
	load = Image.open("Images/rmse_vs_batch.jpg")
	render = ImageTk.PhotoImage(load)
	img = Label(ygn_next_frame, image=render).grid(row=1,column=0,sticky="NEWS",columnspan=3,padx=20,pady=20)
	img.image = render


def ygn_back_page():
	for widget in ygn_next_frame.winfo_children():
		widget.destroy()
		
	yangon_framework()

def pol_next_page():
	for widget in pol_frame.winfo_children():
		widget.destroy()
		
	pol_next_frame = Frame(root)
	pol_next_frame.grid(row=1,column=0,sticky="S")
	#button_frame.place(x=200,y=250,anchor="center")
	#img = ImageTk.PhotoImage(Image.open("/home/dell/Documents/wORKSPACE/GUI_tkinter/rmse_vs_batch.png"))
	
	Label(pol_next_frame,text="Pyin Oo Lwin region:",fg="blue",font=("Helvetica", 22)).grid(row=0,column=0,sticky=W,padx=20,pady=20)
	
	Button(pol_next_frame,text="Back",command=pol_back_page).grid(row=2,column=1,sticky="E",padx=5,pady=15)
	Button(pol_next_frame,text="Home",command=home_page).grid(row=2,column=2,sticky="W",padx=2,pady=15)
	
	load = Image.open("Images/rmse_vs_batch.jpg")
	render = ImageTk.PhotoImage(load)
	img = Label(pol_next_frame, image=render).grid(row=1,column=0,sticky="NEWS",columnspan=3,padx=20,pady=20)
	img.image = render

def pol_back_page():
	for widget in pol_next_frame.winfo_children():
		widget.destroy()
	
	pyin_oo_lwin_framework()
	
def ycc_next_page():
	for widget in ycc_framework.winfo_children():
		widget.destroy()
	
	ycc_next_frame = Frame(root)
	ycc_next_frame.grid(row=1,column=0,sticky="S")
	#button_frame.place(x=200,y=250,anchor="center")
	#img = ImageTk.PhotoImage(Image.open("/home/dell/Documents/wORKSPACE/GUI_tkinter/rmse_vs_batch.png"))
	
	Label(ycc_next_frame,text="Pyin Oo Lwin region:",fg="blue",font=("Helvetica", 22)).grid(row=0,column=0,sticky=W,padx=20,pady=20)
	
	Button(ycc_next_frame,text="Back",command=ycc_back_page).grid(row=2,column=1,sticky="E",padx=5,pady=15)
	Button(ycc_next_frame,text="Home",command=home_page).grid(row=2,column=2,sticky="W",padx=2,pady=15)
	Button(ycc_next_frame,text="Real_Time_Test",command=ycc_real_time_test).grid(row=2,column=3,sticky="E",padx=2,pady=15)
	
	load = Image.open("Images/rmse_vs_batch.jpg")
	render = ImageTk.PhotoImage(load)
	img = Label(ycc_next_frame, image=render).grid(row=1,column=0,sticky="NEWS",columnspan=3,padx=20,pady=20)
	img.image = render

def ycc_back_page():
	for widget in ycc_next_frame.winfo_children():
		widget.destroy()
	
	ycc_framework()

def home_page():
	
	title = Label(top_frame,text="	Weather Prediction Using LSTM	",fg="blue",font=("Helvetica", 22))
	title.grid(row=0,column=0)

	button_frame = Frame(root)
	button_frame.grid(row=1,column=0,sticky="W")

	global radio_region

	radio_region = IntVar()
	Label(button_frame,text=" Choose the region:		",fg="blue",font=("Helvetica", 22)).grid(row=0,column=0,sticky=E,padx=44,pady=24)
	Radiobutton(button_frame,variable=radio_region,text="Yangon Region", value=1).grid(row=1,column=0,sticky=W,padx=44,pady=24)
	Radiobutton(button_frame,variable=radio_region,text="POL Region", value=2).grid(row=2,column=0,sticky=W,padx=44,pady=24)
	Radiobutton(button_frame,variable=radio_region,text="YCC Region", value=3).grid(row=3,column=0,sticky=W,padx=44,pady=24)
	Button(button_frame,text="Process",command=region_selection).grid(row=4,column=1,sticky="E",pady=40)

def yangon_framework():

	for widget in button_frame.winfo_children():
		widget.destroy()
	
	ygn_frame = Frame(root)
	ygn_frame.grid(row=1,column=0,sticky="S")
	#button_frame.place(x=200,y=250,anchor="center")
	#img = ImageTk.PhotoImage(Image.open("/home/dell/Documents/wORKSPACE/GUI_tkinter/rmse_vs_batch.png"))
	
	Label(ygn_frame,text="Yangon region:",fg="blue",font=("Helvetica", 22)).grid(row=0,column=0,sticky=W,padx=20,pady=20)
	
	Button(ygn_frame,text="Next",command=ygn_next_page).grid(row=2,column=1,sticky="E",padx=5,pady=15)
	Button(ygn_frame,text="Home",command=home_page).grid(row=2,column=2,sticky="W",padx=2,pady=15)
	
	load = Image.open("Images/rmse_vs_epoch.jpg")
	render = ImageTk.PhotoImage(load)
	img = Label(ygn_frame, image=render).grid(row=1,column=0,sticky="NEWS",columnspan=3,padx=20,pady=20)
	img.image = render


def pyin_oo_lwin_framework():

	for widget in button_frame.winfo_children():
		widget.destroy()
	
	pol_frame = Frame(root)
	pol_frame.grid(row=1,column=0,sticky="S")
	#button_frame.place(x=200,y=250,anchor="center")
	#img = ImageTk.PhotoImage(Image.open("/home/dell/Documents/wORKSPACE/GUI_tkinter/rmse_vs_batch.png"))
	
	Label(pol_frame,text="Pyin Oo Lwin region:",fg="blue",font=("Helvetica", 22)).grid(row=0,column=0,sticky=W,padx=20,pady=20)
	
	Button(pol_frame,text="Next",command=pol_next_page).grid(row=2,column=1,sticky="E",padx=5,pady=15)
	Button(pol_frame,text="Home",command=home_page).grid(row=2,column=2,sticky="W",padx=2,pady=15)
	
	load = Image.open("Images/rmse_vs_epoch.jpg")
	render = ImageTk.PhotoImage(load)
	img = Label(pol_frame, image=render).grid(row=1,column=0,sticky="NEWS",columnspan=3,padx=20,pady=20)
	img.image = render

def ycc_framework():

	for widget in button_frame.winfo_children():
		widget.destroy()
	
	ycc_frame = Frame(root)
	ycc_frame.grid(row=1,column=0,sticky="S")
	#button_frame.place(x=200,y=250,anchor="center")
	#img = ImageTk.PhotoImage(Image.open("/home/dell/Documents/wORKSPACE/GUI_tkinter/rmse_vs_batch.png"))
	
	Label(ycc_frame,text="YCC region:",fg="blue",font=("Helvetica", 22)).grid(row=0,column=0,sticky=W,padx=20,pady=20)
	
	Button(ycc_frame,text="Next",command=ycc_next_page).grid(row=2,column=1,sticky="E",padx=5,pady=15)
	Button(ycc_frame,text="Home",command=home_page).grid(row=2,column=2,sticky="W",padx=2,pady=15)
	
	load = Image.open("Images/rmse_vs_epoch.jpg")
	render = ImageTk.PhotoImage(load)
	img = Label(ycc_frame, image=render).grid(row=1,column=0,sticky="NEWS",columnspan=3,padx=20,pady=20)
	img.image = render
	
def region_selection():
	
	x = radio_region.get()
	if x == 1:
		yangon_framework()
	elif x ==2:
		pyin_oo_lwin_framework()
	elif x == 3:
		ycc_framework()
	else:
		print("ERROR")


root = Tk()
root.title("Weather Prediction")
root.geometry('+%d+%d' % (350, 125))
root.resizable(0,0)

top_frame = Frame(root,width=700,height=200)
top_frame.grid(row=0,column=0,sticky="NEWS")
#top_frame.place(x=350,y=45,anchor="center")

title = Label(top_frame,text="	Weather Prediction Using LSTM	",fg="blue",font=("Helvetica", 22))
title.grid(row=0,column=0)

button_frame = Frame(root)
button_frame.grid(row=1,column=0,sticky=E)

radio_region = IntVar()
Label(button_frame,text="Choose the region:		",fg="blue",font=("Helvetica", 22)).grid(row=0,column=0,sticky=W,padx=44,pady=24)
Radiobutton(button_frame,variable=radio_region,text="Yangon Region", value=1).grid(row=1,column=0,sticky=W,padx=44,pady=24)
Radiobutton(button_frame,variable=radio_region,text="POL Region", value=2).grid(row=2,column=0,sticky=W,padx=44,pady=24)
Radiobutton(button_frame,variable=radio_region,text="YCC Region", value=3).grid(row=3,column=0,sticky=W,padx=44,pady=24)
Button(button_frame,text="Process",command=region_selection).grid(row=4,column=1,sticky="E",pady=40)

root.mainloop()
