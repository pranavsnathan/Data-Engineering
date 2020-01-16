from flask import Flask,render_template,url_for,request
import pandas as pd
import math
import pickle
import numpy as np


app = Flask(__name__)

@app.route('/')
def home():
	return render_template('home.html')

@app.route('/predict',methods=['POST'])
def predict():

    rf = open("RFFinalModel.pkl", "rb")
    rf_reg = pickle.load(rf)

    if request.method == 'POST':
        tenure = request.form['tenure']
        paperlessbilling = request.form['paperlessbilling']
        monthlycharges = request.form['monthlycharges']
        totalcharges = request.form['totalcharges']
        fiberoptic = request.form['fiberoptic']
        onlinesecurity = request.form['onlinesecurity']
        techsupport = request.form['techsupport']
        monthlycontract = request.form['monthlycontract']
        twoyearcontract = request.form['twoyearcontract']
        echequepayment = request.form['echequepayment']
        print(echequepayment)

        tenure = int(tenure)
        monthlycharges = float(monthlycharges)
        paperlessbilling = int(paperlessbilling)
        totalcharges = float(totalcharges)
        fiberoptic = int(fiberoptic)
        onlinesecurity = int(onlinesecurity)
        techsupport = int(techsupport)
        monthlycontract = int(monthlycontract)
        twoyearcontract = int(twoyearcontract)
        echequepayment = int(echequepayment)

        data = []
        data = [tenure, paperlessbilling, monthlycharges,totalcharges, fiberoptic, onlinesecurity, techsupport, monthlycontract, twoyearcontract, echequepayment]
        my_prediction = math.floor(rf_reg.predict(pd.DataFrame(data).T))
        my_prediction = int(my_prediction)
        my_prediction = np.abs(my_prediction)

    return render_template('result.html',prediction = my_prediction)

if __name__ == '__main__':
	app.run()