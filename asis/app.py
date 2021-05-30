# -*- coding: utf-8 -*-
"""
Created on Wed Mar 24 11:12:40 2021

@author: bawej
"""

import numpy as np
from flask import Flask, request, jsonify, render_template
import pickle

app = Flask(__name__)
model = pickle.load(open('model.pkl', 'rb'))

@app.route('/')
def home():
    return render_template('index.html')

######################################################################################################
@app.route('/about') 
def about():                                                 #CHANGES MADE ONLY HERE
    return render_template('about.html')
######################################################################################################

@app.route("/practice")
def practice():
    """ return the rendered template """
    return render_template("practice.html")

#######################################################################################################
def check_user_input(input):
    try:
        # Convert it into integer
        val = int(input)
        
    except ValueError:
        try:                                                #BLOCK CAN BE REMOVED
            # Convert it into float
            val = float(input)
            
        except ValueError:
            val=str(input)
    return val
########################################################################################################
@app.route('/predict',methods=['POST'])
def predict():
    '''
    For rendering results on HTML GUI
    '''
    l=[]
    
    for x in request.form.values() :
         z=x
         if isinstance(check_user_input(x),int)== True:                                      #REMOVE CHECK_USER_INPUT()
                x=z
         elif isinstance(check_user_input(x),str)== True:                                    #REMOVE CHECK_USER_INPUT()
               if x == 'yes':
                    x=1
               elif x=='no':
                    x=0
               elif x=='regular':
                    x=2
               elif x == 'irregular':
                    x=4
##################################################################################################################
               else:
                   return render_template('practice.html', prediction_text='Kindly enter valid input ')
         else:                                                                                            #BLOCK CAN BE REMOVED
               return render_template('practice.html', prediction_text='Kindly enter valid input ')
###################################################################################################################        
         l.append(int(x))    
    
      
    final_features = [np.array(l)]
    prediction = model.predict(final_features)

    output = round(prediction[0], 2)
    if output == 1:
        output ='Yes'
        
    elif output == 0:
        output ='No'

    return render_template('practice.html', prediction_text='Do I have PCOS ? {}'.format(output))

@app.route('/predict_api',methods=['POST'])
def predict_api():
    '''
    For direct API calls trought request
    '''
    data = request.get_json(force=True)
    prediction = model.predict([np.array(list(data.values()))])

    output = prediction[0]
    return jsonify(output)

if __name__ == "__main__":
    app.run(debug=True)