import numpy as np
import pandas as pd
from flask import Flask, request, render_template
import pickle

app = Flask(__name__)
model = pickle.load(open('model.pkl', 'rb'))

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict',methods=['POST'])
def predict():
    input_features = [float(x) for x in request.form.values()]
    features_value = [np.array(input_features)]
    
    features_name = [ "age", "anaemia", "creatinine_phosphokinase", "diabetes", "ejection_fraction", "high_blood_pressure", "platelets",                             "serum_creatinine", "serum_sodium", "sex", "smoking", "time" ]
    
    df = pd.DataFrame(features_value, columns=features_name)
    output = model.predict(df)
        
    if output == 1:
        res_val = "** HEART FAILURE **"
    else:
        res_val = " NO HEART FAILURE "
        

    return render_template('index.html', prediction_text='THE PATIENT IS LIKELY TO HAVE {}'.format(res_val))

if __name__ == "__main__":
    app.run()