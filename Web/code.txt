import tkinter as tk
from tkinter import Message ,Text
from PIL import Image, ImageTk
import pandas as pd
from tkinter import *

import tkinter.ttk as ttk
import tkinter.font as font
import tkinter.messagebox as tm
import matplotlib.pyplot as plt
import csv
import numpy as np
from PIL import Image, ImageTk
from tkinter import filedialog
import tkinter.messagebox as tm

import DTALG as DT
import LogisticRegression as LR
import svmalgo as svm
import XGboostalgo as xgbalgo
import RandomForest as RF
import Predict as pred

bgcolor="#DAF7A6"
bgcolor1="#B7C526"
fgcolor="black"

  



def Home():
	global window
	def clear():
	    print("Clear1")
	      
	    txt1.delete(0, 'end')
	    txt2.delete(0, 'end')
	    txt3.delete(0, 'end')
	    txt4.delete(0, 'end')    
	    txt5.delete(0, 'end')
	    txt6.delete(0, 'end')
	    txt7.delete(0, 'end')
	    txt8.delete(0, 'end')    
	    txt9.delete(0, 'end')
	    txt10.delete(0, 'end')
	   
	    

	window = tk.Tk()
	
	window.title("Heart Disease Prediction")
	 
	window.geometry('1280x720')
	window.configure(background=bgcolor)
	#window.attributes('-fullscreen', True)

	window.grid_rowconfigure(0, weight=1)
	window.grid_columnconfigure(0, weight=1)
	

	message1 = tk.Label(window, text="Early Detection of Cardiac Arrest in Newborn Babies" ,bg=bgcolor  ,fg=fgcolor  ,width=50  ,height=2,font=('times', 30, 'italic bold underline')) 
	message1.place(x=100, y=1)

	

	lbl1 = tk.Label(window, text="Enter Age in months",width=20  ,height=2  ,fg=fgcolor  ,bg=bgcolor ,font=('times', 15, ' bold ') ) 
	lbl1.place(x=50, y=100)
	
	txt1 = tk.Entry(window,width=20,bg="white" ,fg="black",font=('times', 15, ' bold '))
	txt1.place(x=300, y=115)

	lbl2 = tk.Label(window, text="Weight",width=20  ,height=2  ,fg=fgcolor  ,bg=bgcolor ,font=('times', 15, ' bold ') ) 
	lbl2.place(x=50, y=170)
	
	txt2 = tk.Entry(window,width=20,bg="white" ,fg="black",font=('times', 15, ' bold '))
	txt2.place(x=300, y=175)
	lbl3 = tk.Label(window, text="Height ",width=20  ,height=2  ,fg=fgcolor  ,bg=bgcolor ,font=('times', 15, ' bold ') ) 
	lbl3.place(x=50, y=230)
	
	txt3 = tk.Entry(window,width=20,bg="white" ,fg="black",font=('times', 15, ' bold '))
	txt3.place(x=300, y=235)
	
	lbl4 = tk.Label(window, text="Gender",width=20  ,height=2  ,fg=fgcolor  ,bg=bgcolor ,font=('times', 15, ' bold ') ) 
	lbl4.place(x=50, y=285)
	
	txt4 = tk.Entry(window,width=20,bg="white" ,fg="black",font=('times', 15, ' bold '))
	txt4.place(x=300, y=290)

	lbl5 = tk.Label(window, text="Heart Rate ",width=20  ,height=2  ,fg=fgcolor  ,bg=bgcolor ,font=('times', 15, ' bold ') ) 
	lbl5.place(x=50, y=350)
	
	txt5 = tk.Entry(window,width=20,bg="white" ,fg="black",font=('times', 15, ' bold '))
	txt5.place(x=300, y=365)
	lbl6 = tk.Label(window, text="Oxygen Saturation ",width=20  ,height=2  ,fg=fgcolor  ,bg=bgcolor ,font=('times', 15, ' bold ') ) 
	lbl6.place(x=50, y=420)
	
	txt6 = tk.Entry(window,width=20,bg="white" ,fg="black",font=('times', 15, ' bold '))
	txt6.place(x=300, y=435)
	lbl7 = tk.Label(window, text="Respiratory Rate",width=20  ,height=2  ,fg=fgcolor  ,bg=bgcolor ,font=('times', 15, ' bold ') ) 
	lbl7.place(x=550, y=100)
	
	txt7 = tk.Entry(window,width=20,bg="white" ,fg="black",font=('times', 15, ' bold '))
	txt7.place(x=800, y=115)

	lbl8 = tk.Label(window, text="Systolic Blood Pressure",width=20  ,height=2  ,fg=fgcolor  ,bg=bgcolor ,font=('times', 15, ' bold ') ) 
	lbl8.place(x=550, y=170)
	
	txt8 = tk.Entry(window,width=20,bg="white" ,fg="black",font=('times', 15, ' bold '))
	txt8.place(x=800, y=175)
	lbl9 = tk.Label(window, text="Diastolic Blood Pressure ",width=20  ,height=2  ,fg=fgcolor  ,bg=bgcolor ,font=('times', 15, ' bold ') ) 
	lbl9.place(x=550, y=230)
	
	txt9 = tk.Entry(window,width=20,bg="white" ,fg="black",font=('times', 15, ' bold '))
	txt9.place(x=800, y=235)
	
	lbl10 = tk.Label(window, text="Mean Blood Pressure",width=20  ,height=2  ,fg=fgcolor  ,bg=bgcolor ,font=('times', 15, ' bold ') ) 
	lbl10.place(x=550, y=285)
	
	txt10 = tk.Entry(window,width=20,bg="white" ,fg="black",font=('times', 15, ' bold '))
	txt10.place(x=800, y=290)

	



		

	def LRprocess():
		sym="Child_Heart_Stage_dataset.csv"
		if sym != "":
			LR.process(sym)
			tm.showinfo("Input", "Logistic Regression Successfully Finished")
		else:
			tm.showinfo("Input error", "Select Dataset File")

	def RFprocess():
		sym="Child_Heart_Stage_dataset.csv"
		if sym != "":
			RF.process(sym)
			tm.showinfo("Input", "Random Forest Successfully Finished")
		else:
			tm.showinfo("Input error", "Select Dataset File")

	def DTProcess():
		sym="Child_Heart_Stage_dataset.csv"
		if sym != "":
			DT.process(sym)
			tm.showinfo("Input", "Decision Tree Successfully Finished")
		else:
			tm.showinfo("Input error", "Select Dataset File")

	def svmprocess():
		sym="Child_Heart_Stage_dataset.csv"
		if sym != "":
			svm.process(sym)
			tm.showinfo("Input", "SVM Finished Success")
		else:
			tm.showinfo("Input error", "Select Dataset File")
	def xgbProcess():
		sym="Child_Heart_Stage_dataset.csv"
		if sym != "":
			xgbalgo.process(sym)
			tm.showinfo("Input", "XgBoost Finished Success")
		else:
			tm.showinfo("Input error", "Select Dataset File")

	def Predictprocess():
                sym="Child_Heart_Stage_dataset.csv"
                a1=txt1.get()
                a2=txt2.get()
                a3=txt3.get()
                a4=txt4.get()
                if a4=="male":
                        a4=1
                else:
                        a4=0
                a5=txt5.get()
                a6=txt6.get()
                a7=txt7.get()
                a8=txt8.get()
                a9=txt9.get()
                a10=txt10.get()
                if a1 == "":
                        tm.showinfo("Insert error", "Enter Age")
                elif a2 == "":
                        tm.showinfo("Insert error", "Enter Weight")
                elif a3 == "":
                        tm.showinfo("Insert error", "Enter Height")
                elif a4 == "":
                        tm.showinfo("Insert error", "Enter Gender")
                elif a5 == "":
                        tm.showinfo("Insert error", "Enter Heart Rate")
                elif a6 == "":
                        tm.showinfo("Insert error", "Enter oxygen saturation")
                elif a7 == "":
                        tm.showinfo("Insert error", "Enter Respiratory Rate")
                elif a8 == "":
                        tm.showinfo("Insert error", "Enter Systolic Blood Pressure")
                elif a9 == "":
                        tm.showinfo("Insert error", "Enter Diastolic Blood Pressure")
                elif a10=="":
                        tm.showinfo("Insert error", "Enter Mean Blood Pressure")
                else:
                        print("a1==",a1)
                        print("a2==",a2)
                        print("a3==",a3)
                        print("a4==",a4)
                        print("a5==",a5)
                        print("a6==",a6)
                        print("a7==",a7)
                        print("a8==",a8)
                        print("a9==",a9)
                        print("a10==",a10)
                        
                        res,treat = pred.process(sym,float(a1),float(a2),float(a3),float(a4),float(a5),float(a6),float(a7),float(a8),float(a9),float(a10))
                        print(res)
                        if res!="":
                                tm.showinfo("Result", str(res))
                                tm.showinfo("Recommended Treatment", str(treat))
                        else:
                                tm.showinfo("Input error", "Select Dataset File")
                        
                                

	

	clearButton = tk.Button(window, text="Clear", command=clear  ,fg=fgcolor  ,bg=bgcolor1  ,width=14  ,height=2 ,activebackground = "Red" ,font=('times', 15, ' bold '))
	clearButton.place(x=10, y=500)
	 
	proc = tk.Button(window, text="Random Forest", command=RFprocess  ,fg=fgcolor   ,bg=bgcolor1   ,width=14  ,height=2, activebackground = "Red" ,font=('times', 15, ' bold '))
	proc.place(x=210, y=500)
	

	LRbutton = tk.Button(window, text="SVM", command=svmprocess  ,fg=fgcolor   ,bg=bgcolor1   ,width=14  ,height=2, activebackground = "Red" ,font=('times', 15, ' bold '))
	LRbutton.place(x=410, y=500)


	RFbutton = tk.Button(window, text="XG Boost", command=xgbProcess  ,fg=fgcolor   ,bg=bgcolor1 ,width=14  ,height=2, activebackground = "Red" ,font=('times', 15, ' bold '))
	RFbutton.place(x=610, y=500)

	

	PRbutton = tk.Button(window, text="Predict", command=Predictprocess  ,fg=fgcolor   ,bg=bgcolor1   ,width=14  ,height=2, activebackground = "Red" ,font=('times', 15, ' bold '))
	PRbutton.place(x=810, y=500)

	quitWindow = tk.Button(window, text="Quit", command=window.destroy  ,fg=fgcolor   ,bg=bgcolor1  ,width=15  ,height=2, activebackground = "Red" ,font=('times', 15, ' bold '))
	quitWindow.place(x=1010, y=500)

	window.mainloop()
Home()

