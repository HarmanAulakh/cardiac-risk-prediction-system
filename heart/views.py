from django.shortcuts import render, redirect
from django.http import HttpResponse
from django.conf import settings
from django.db.models import Q
from django.core.files.storage import FileSystemStorage
from django.contrib import messages
from django.db import connection
import os
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from statsmodels.tsa.arima_model import ARIMA
from sklearn.metrics import mean_squared_error
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
import json
import log


# Create your views here.

def details(request, fileId):
    ### Get the File name ###
    fileDetails = getFileData(fileId)
    
    #### Read CSV File and Upload into Database ####
    print(os.getcwd())
    csv_data = pd.read_csv(str(os.getcwd())+'/media/'+ fileDetails['files_original_file_name'], header=0, encoding = 'unicode_escape')
    csv_data = csv_data.values.tolist()

    context = {
        "heartlist": csv_data
    }

    # Message according medicines Role #
    context['heading'] = "Heart Details"
    return render(request, 'heart-list.html', context)

def prediction(request, fileId):
    context = {}
    ### Get the Database Configuration ####
    if (request.method == "POST"):
        ### Insert the File Details #####
        form_data = [[
            request.POST['age'],
            request.POST['gender'], 
            request.POST['chest_pain_type'], 
            request.POST['bp'],
            request.POST['sc'], 
            request.POST['sugar'], 
            request.POST['resting'], 
            request.POST['hra'], 
            request.POST['eia'], 
            request.POST['depression'], 
            request.POST['st'], 
            request.POST['vessels'], 
            request.POST['thal']
        ]]
        prediction = dataTraining(fileId, form_data)
        
        pcontext = {
            "prediction": prediction,
            "form_data": (form_data[0])
        }
        return render(request, 'prediction-result.html', pcontext)
    # Message according medicines Role #
    context['heading'] = "Heart Details"
    return render(request, 'prediction.html', context)
    
def listToString(list):
    lst=[]
    for i in list:
        lst.append(i[0])
    return lst

def cumSumToString(list):
    lst=[]
    for i in list:
        lst.append(i)
    return lst
    
def dictfetchall(cursor):
    "Return all rows from a cursor as a dict"
    cursor.execute("SELECT * FROM files WHERE user_file_id = " + str(log.loggedInUser))
    columns = [col[0] for col in cursor.description]
    return [
        dict(zip(columns, row))
        for row in cursor.fetchall()
    ]

def getFileData(loggedInUser):
    cursor = connection.cursor()
    cursor.execute("SELECT * FROM files WHERE user_file_id = " + str(log.loggedInUser))
    dataList = dictfetchall(cursor)
    return dataList[0]

def dataTraining(fileId, testData):
    ### Get the File name ###
    fileDetails = getFileData(fileId)
    
    #### Read CSV File and Upload into Database ####
    dataset = pd.read_csv(str(os.getcwd())+'/media/'+ fileDetails['files_original_file_name'])
    
    info = [
        "age",
        "1: male, 0: female",
        "chest pain type, 1: typical angina, 2: atypical angina, 3: non-anginal pain, 4: asymptomatic",
        "resting blood pressure",
        "serum cholestoral in mg/dl",
        "fasting blood sugar > 120 mg/dl",
        "resting electrocardiographic results (values 0,1,2)",
        "maximum heart rate achieved",
        "exercise induced angina",
        "oldpeak = ST depression induced by exercise relative to rest",
        "the slope of the peak exercise ST segment",
        "number of major vessels (0-3) colored by flourosopy",
        "thal: 3 = normal; 6 = fixed defect; 7 = reversable defect"
    ]

        
    # Removing target varaible 
    predictors = dataset.drop("target",axis=1)
    target = dataset["target"]

    # Deviding dataset for training and testing
    X_train,X_test,Y_train,Y_test = train_test_split(predictors,target,test_size=0.20,random_state=0)

    # Training using Logistic Regression
    lr = LogisticRegression()
    lr.fit(X_train,Y_train)
    Y_pred_lr = lr.predict(X_test)
    score_lr = round(accuracy_score(Y_pred_lr,Y_test)*100,2)

    # Form data 
    form_data = pd.DataFrame(testData)

    test_data_set = dataset.tail(5)
    test_data_set = test_data_set.drop(columns=["target"])
    prediction = lr.predict(form_data)
    return prediction


def smape_kun(y_true, y_pred):
    return np.mean((np.abs(y_pred - y_true) * 200/ (np.abs(y_pred) + np.abs(y_true))))