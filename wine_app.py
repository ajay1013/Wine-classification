"""
Created on Thu Dec 10 03:23:57 2020

@author: Ajay Kumar
"""


import pickle
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from flask import Flask, request, render_template



app = Flask(__name__)

model = pickle.load(open('classifier_wine.pkl', 'rb'))

@app.route('/')
def home():
    return render_template('Wine Classification.html')

@app.route('/predict',methods=['POST'])
def predict():
    int_features = [float(x) for x in request.form.values()]
    features = [np.array(int_features)]
    feat = pd.DataFrame(data=features, columns=["fixed acidity","volatile acidity","citric acid","residual sugar","chlorides","free sulfur dioxide","total sulfur dioxide","density","pH","sulphates","alcohol","quality"])
    data3 = pd.DataFrame(StandardScaler().fit_transform(feat))
    prediction = model.predict(data3)
    if prediction==1:
        return render_template('Wine Classification.html', prediction_text='This is a RED wine')
    else:
        return render_template('Wine Classification.html', prediction_text='This is a WHITE wine')

if __name__ == "__main__":
    app.run(port=12000)
