from flask import Flask,render_template, request, jsonify
import flask
import numpy as np
import traceback
import pickle
import pandas as pd
import os
 
 
# App definition
app = Flask(__name__,template_folder='templates')

port = int(os.environ.get("PORT",5000))
 
# importing models
with open(os.path.join(os.getcwd(),'model','model.pkl'), 'rb') as f:
   classifier = pickle.load (f)
 
with open(os.path.join(os.getcwd(),'model','model_columns.pkl'), 'rb') as f:
   model_columns = pickle.load (f)
 
 
@app.route('/')
def welcome():
   return "Recommendation System"
 
@app.route('/predict', methods=['POST','GET'])
def predict():
  
   if flask.request.method == 'GET':
       return "Prediction page"
 
   if flask.request.method == 'POST':
       try:
           json_ = request.json
           print(json_)
           query_ = pd.get_dummies(pd.DataFrame(json_,index=[0]))
           query = query_.reindex(columns = model_columns, fill_value= 0)
           prediction = list(classifier.predict(query))
           '''if prediction == [0]:
               output = "Not Fraud"
           else:
               output = "Fraud"'''
 
           return jsonify({
               "prediction":str(prediction)
           })
 
       except:
           return jsonify({
               "trace": traceback.format_exc()
               })
      
#Driver Code 
if __name__ == "__main__":
   app.run(debug=True,host='0.0.0.0',port=port)
