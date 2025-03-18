"""
This script handles the execution of the Flask Web Server(Web Application + JSON API)
"""

from flask import Flask, render_template, request, redirect, url_for, session, jsonify, flash
#from flaskext.mysql import MySQL
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split
from sklearn import tree 
from sklearn import model_selection
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import cross_val_score
# from googleplaces import GooglePlaces, types, lang 
from flask_socketio import SocketIO
import pandas as pd 
import numpy as np
import pickle
import re
import os
import random
import hashlib 
#import bcrypt
import json
import pybase64
from datetime import date
from sklearn.preprocessing import normalize
import MySQLdb
from datetime import timedelta
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
#import cv2
import seaborn as sns
import scipy.stats as stats
import sklearn
import os
import pickle
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import normalize
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn import preprocessing
import xgboost as xgb
from werkzeug.utils import secure_filename

import Predict as pred
UPLOAD_FOLDER = './static/input'
app = Flask(__name__)

port = int(os.environ.get('PORT', 5000))


# Change this to your secret key (can be anything, it's for extra protection)
app.secret_key = 'canada$God7972#'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
# Enter your database connection details below
app.config['MYSQL_DATABASE_HOST'] = 'localhost'
app.config['MYSQL_DATABASE_USER'] = 'root'
app.config['MYSQL_DATABASE_PASSWORD	'] ="root"
app.config['MYSQL_DATABASE_DB'] = 'cardio'

# Intialize MySQL
# mysql = MySQL(autocommit=True)
# mysql.init_app(app)
mydb = MySQLdb.connect(host='localhost',user='root',passwd='root',db='cardio')
#app.permanent_session_lifetime = timedelta(minutes=15)

#ecg = ECG()

#Homepage
@app.route('/')
def index():
    if 'loggedin' not in session:
        return render_template('index.html')
    else:
        return home()

#Dashboard
@app.route('/dashboard')
def home():
    # Check if user is loggedin
    print("session===22",session)
    if 'loggedin' in session:
        print("Inside If in dashbord")
        # User is loggedin show them the home page
        cursor = mydb.cursor()
        cursor.execute('SELECT * FROM doctors WHERE ID = %s', (session['id'],))
        account = cursor.fetchone()
        
        print("is doctor==",session['isdoctor'])
        return render_template('dashboard.html', account = account, isdoctor=session['isdoctor'])
    # User is not loggedin redirect to login page
    return redirect(url_for('login'))




#Doctor Register
@app.route('/docregister', methods=['GET', 'POST'])
def docregister():
    if 'loggedin' not in session:
    # Output message if something goes wrong...
        msg = ''
        # Check if "username", "password" and "email" POST requests exist (user submitted form)
        if request.method == 'POST' and 'username' in request.form and 'password' in request.form and 'email' in request.form:
            # Create variables for easy access
            username = request.form['username']
            password = request.form['password']
            email = request.form['email']
            full_name = request.form['full_name']
            registration_number = request.form['registration_number']
            contact_number = request.form['contact_number']
            spec = request.form['specialization']
            address = request.form['address']
            if(username and password and email and full_name and registration_number and contact_number and spec and address):
            # Check if account exists using MySQL
                cursor = mydb.cursor()
                cursor.execute('SELECT * FROM doctors WHERE Username = %s', (username,))
                account = cursor.fetchone()
                # If account exists show error and validation checks
                if account:
                    msg = 'Account already exists!'
                    flash(msg)
                elif not re.match(r'[^@]+@[^@]+\.[^@]+', email):
                    msg = 'Invalid email address!'
                    flash(msg)
                elif not re.match(r'[A-Za-z0-9]+', username):
                    msg = 'Username must contain only characters and numbers!'
                    flash(msg)
                else:
                    # Account doesnt exists and the form data is valid, now insert new account into users table
                    #hashed_password = bcrypt.hashpw(password.encode('utf-8'), bcrypt.gensalt())
                    cursor.execute('INSERT INTO doctors VALUES (NULL, %s, %s, %s, %s, %s, %s, %s, %s, %s)', ( username, password, email, full_name, registration_number, contact_number, "Default Hospital" , spec, address ))
                    mydb.commit()
                    msg = 'You have successfully registered!'
                    cursor.execute('SELECT * FROM doctors WHERE Username = %s', (username,))
                    # Fetch one record and return result
                    account = cursor.fetchone()
                    session['loggedin'] = True
                    session['id'] = account[0]
                    session['username'] = account[1]
                    session['isdoctor'] = 1
                    return home()
            else:
                msg = 'Please fill out the form!'
                flash(msg)
        elif request.method == 'POST':
            # Form is empty... (no POST data)
            msg = 'Please fill out the form!'
    else:
        return home()
    # Show registration form with message (if any)
    return render_template('doctorlogin.html', msg=msg)

#Doctor Login
@app.route('/doclogin', methods=['GET', 'POST'])
def doclogin():
    if 'loggedin' not in session:
    # Output message if something goes wrong...
        msg = ''
        # Check if "username" and "password" POST requests exist (user submitted form)
        if request.method == 'POST' and 'username' in request.form and 'password' in request.form:
            # Create variables for easy access
            username = request.form['username']
            password = request.form['password']
            if(username and password):

                # Check if account exists using MySQL
                cursor = mydb.cursor()
                cursor.execute('SELECT * FROM doctors WHERE Username = %s', (username,))
                # Fetch one record and return result
                account = cursor.fetchone()
                # If account exists in accounts table in out database
                if account:
                    if password==account[2]:
                        # Create session data, we can access this data in other routes
                        session['loggedin'] = True
                        session['id'] = account[0]
                        session['username'] = account[1]
                        session['isdoctor'] = 1
                        # Redirect to home page
                        print("session==",session)
                        return home()
                    else:
                        # Account doesnt exist or username/password incorrect
                        msg = 'Incorrect username/password!'
                        flash(msg)
                else:
                    # Account doesnt exist or username/password incorrect
                    msg = 'Incorrect username/password!'
                    flash(msg)
            else:
                msg = 'Please provide both username and password!'
                flash(msg)
    else:
        return home()
    # Show the login form with message (if any)
    return render_template('doctorlogin.html', msg=msg)


# Diagnose Based on the Cardiovascular problems
@app.route('/diagnosecardio',methods=['GET','POST'])
def diagnosecardio():
    # Check if user is loggedin
    if 'loggedin' in session:
        cursor = mydb.cursor()
        if session["isdoctor"]:
            cursor.execute('SELECT * FROM doctors WHERE ID = %s', (session['id'],))
        account = cursor.fetchone()
        
        

        
        if(request.method == 'POST'):
            a1 = request.form['age']
            a2 = request.form['weight']
            a3= request.form['height'] # in kilograms
            a4= request.form['gen']
            a5=request.form['hr']
            a6= request.form['os']
            a7= request.form['rr'] 
            a8= request.form['Sys']
           # Systolic blood pressure
            a9= request.form['Dys'] # Diastolic blood pressure
            a10= request.form['mbp'] # 1: normal, 2: above normal, 3: well above normal
            data=pd.read_csv("./Dataset/Child_Heart_Stage_dataset.csv")
            label_encoder = preprocessing.LabelEncoder()
            data['Diagnosis']= label_encoder.fit_transform(data['Diagnosis'])
            data['Gen']= label_encoder.fit_transform(data['Genero'])
            X=data[['Age', 'Weight (Kg)', 'Height (cms)', 'Gen','Heart Rate', 'oxygen saturation', 'Respiratory Rate','Systolic Blood Pressure', 'Diastolic Blood Pressure','Mean Blood Pressure']].values
            y=data['Diagnosis'].values
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
            xgb_clf = xgb.XGBClassifier(objective='multi:softmax', num_class=3, missing=1, early_stopping_rounds=10, eval_metric=['merror','mlogloss'],seed=42)
            xgb_clf.fit(X_train, y_train, verbose=0, eval_set=[(X_train, y_train), (X_test, y_test)])
            int_features= [float(x) for x in request.form.values()]
            print("Len of Xtest==",len(int_features))
            final4=[np.array(int_features)]
            #x_test=final4.reshape(1,-1)
            y_pred=xgb_clf.predict(final4)
            res=y_pred[0]
            result=""
            treat=""
            predres=""
            if y_pred[0]==0:
                result="Stage Normal"
                treat="dexrazoxane is no longer contraindicated"
            elif y_pred[0]==1:
                result="Stage Mild"
                treat="Adeno-associated virus gene therapy"
            elif y_pred[0]==2:
                result="Stage Moderate"
                treat="anti–interleukin-6 receptor antagonist such as tocilizumab "
            elif y_pred[0]==3:
                result="Stage Severe"
                treat="Immediate surgey need to given"
            else:
                result="No Disease"
                treat="U can discharge from ICU to Gengeral Ward"


                      

            #res,treat=pred.process("",float(a1),float(a2),float(a3),float(a4),float(a5),float(a6),float(a7),float(a8),float(a9),float(a10))

           

            
            

            return render_template('cardioanswer.html',ans=result,treat=treat,account=account)
        else:
            return render_template('cardiodetails.html',account=account)
    # User is not loggedin redirect to login page
    return redirect(url_for('login'))


# Account information visible inside dashboard
@app.route('/myaccount')
def myaccount():
    if 'loggedin' in session:
        cursor = mydb.cursor()
        if session["isdoctor"]:
            cursor.execute('SELECT * FROM doctors WHERE ID = %s', (session['id'],))
        else:
            cursor.execute('SELECT * FROM users WHERE ID = %s', (session['id'],))
        account = cursor.fetchone()
        return render_template('myaccount.html', account=account, isDoctor = session["isdoctor"])
    else:
        return redirect(url_for('login'))



"""
Code for the Chat App
which is based on Sockets.io
"""

socketio = SocketIO(app)


# http://localhost:5000/logout - this will be the logout page
@app.route('/logout')
def logout():
   # Remove session data, this will log the user out
   session.pop('loggedin', None)
   session.pop('id', None)
   session.pop('username', None)
   # Redirect to login page
   return redirect(url_for('index'))

#run the Flask Server
if __name__ == '__main__':
	socketio.run(app, debug=True)
    
"""-------------------------------End of Web Application-------------------------------"""
