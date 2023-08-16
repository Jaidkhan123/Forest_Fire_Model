from flask import Flask, request, jsonify, render_template
# jsonify to return object in json format
import pickle
import numpy as np 
import pandas as pd
from sklearn.preprocessing import StandardScaler

# import Ridge Regressor Model and StandardScaler pickle file
ridge_model = pickle.load(open('Models/ridge.pkl','rb'))
scaler_model = pickle.load(open('Models/scaler.pkl','rb'))

app = Flask(__name__)

# Route for Home Page    
# With every route we have to write a function
@app.route('/')
def index():
    return render_template('index.html')  # render_template will see template folder in the workspace

@app.route('/predictdata', methods= ['GET', 'POST'])  # we have to handle both get and post data 
def home():
    if request.method == "POST":
        Temperature=float(request.form.get('Temperature'))
        RH = float(request.form.get('RH'))
        Ws = float(request.form.get('Ws'))
        Rain = float(request.form.get('Rain'))
        FFMC = float(request.form.get('FFMC'))
        DMC = float(request.form.get('DMC'))
        ISI = float(request.form.get('ISI'))
        Classes = float(request.form.get('Classes'))
        Region = float(request.form.get('Region'))

        new_data_scaled=scaler_model.transform([[Temperature,RH,Ws,Rain,FFMC,DMC,ISI,Classes,Region]])
        result=ridge_model.predict(new_data_scaled)  
        # it gives result in list format 

        return render_template('home.html',result=result[0])

    else : 
        return render_template('home.html')


# @app.route("/")
# def hello_world():
#     return "<h1>Hello, World!</h1>"

if __name__=="__main__":
    app.run(host="0.0.0.0")  # ye jha bhi run ho rha hota hai ye local address k sath map kr leta hai that is 
                             # 127.0.0.1 
    # app.run(host="0.0.0.0", port = ???) we can give port here also instead of 5000
