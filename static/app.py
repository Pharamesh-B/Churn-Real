import tensorflow as tf
from keras.models import Sequential,load_model,model_from_json
from keras.layers import Dense, Dropout,Activation,MaxPooling2D,Conv2D,Flatten
from keras.applications.imagenet_utils import preprocess_input, decode_predictions
from keras.preprocessing.image import load_img
from keras.preprocessing import image
import numpy as np
import h5py
import os
import sys
import json
from sklearn.preprocessing import StandardScaler
from predictor import sc
import pandas as pd
# Flask utils
from flask import Flask, make_response, redirect, url_for, request, render_template
from werkzeug.utils import secure_filename
from flask import send_file
from io import BytesIO


# Define a flask app
app = Flask(__name__)

with open('customer_churn_prediction_model.json','r') as f:
    model = model_from_json(f.read())


# Load your trained model
model.load_weights('customer_churn_prediction_model.h5')   

 


@app.route('/')
@app.route('/first') 
def first():
	return render_template('first.html')
@app.route('/login')
def login():
	return render_template('login.html')
 

@app.route('/performance') 
def performance():
	return render_template('performance.html')    
@app.route('/upload') 
def upload():
	return render_template('upload.html') 
@app.route('/preview',methods=["POST"])
def preview():
    if request.method == 'POST':
        dataset = request.files['datasetfile']
        df = pd.read_csv(dataset,encoding = 'unicode_escape')
        df.set_index('Id', inplace=True)
        return render_template("preview.html",df_view = df)  

@app.route('/prediction', methods = ['GET', 'POST'])
def prediction():
    # Main page
    return render_template('prediction.html')


@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        # Get the values from the form 
        credit_score = request.form['cr_score']
        age = request.form['age']
        tenure = request.form['tenure']
        balance = request.form.get('balance')
        number_of_products = request.form.get('no_of_products')
        estimated_salary = request.form['salary']
        country = request.form['country']
        gender = request.form['gender']
        has_credit_card = request.form['cr_card']
        is_active_member = request.form['active_member']
        print([credit_score,age,tenure,balance,number_of_products,estimated_salary,country,gender,has_credit_card,is_active_member])
        # Process input 
        if country=="France":
            countries= [0,0]
        elif country=="Germany":
            countries = [1,0]
        else:
            countries = [0,1]
        # Make Prediction
        prediction = model.predict(sc.transform(np.array([[countries[0],countries[1],credit_score,gender,age,tenure,balance,number_of_products,has_credit_card,is_active_member,estimated_salary]])))
        # Process your result for human
        if prediction > 0.5:
            result = "The customer will leave the bank"
        else:
            result = "The customer won't leave the bank"
        return render_template('prediction.html', prediction_value=result)
     
@app.route('/path') 
def path():
	return render_template('path.html')

@app.route('/upload_excel', methods=['POST'])
def upload_excel():
    if request.method == 'POST':
        # Get the uploaded file
        uploaded_file = request.files['excel_file']

        # Check if the file is an Excel file
        if uploaded_file.filename.endswith(('.xls', '.xlsx')):
            # Load the Excel file into a DataFrame
            #df = pd.read_excel(uploaded_file)
            df = pd.read_excel(uploaded_file, engine='openpyxl')
            # Perform prediction for each row in the DataFrame
            df['Prediction'] = df.apply(predict_row, axis=1)

            # Export the modified DataFrame to Excel
            output = BytesIO()
            writer = pd.ExcelWriter(output, engine='xlsxwriter')
            df.to_excel(writer, index=False)
            writer.close()  # Save the ExcelWriter object, not XlsxWriter object
            output.seek(0)

            # Return the modified Excel file as a download
            return send_file(output, attachment_filename='predicted_results.xlsx', as_attachment=True)
            # return render_template('prediction.html', prediction_file='predicted_results.xlsx')
        else:
            return "Please upload a valid Excel file"

    return render_template('prediction.html')
    # return render_template('prediction.html', prediction_file=None)

def predict_row(row):
    # Process input
    country = row['Geography']
    countries = [0, 0]  # Default values for 'France'
    if country == "Germany":
        countries = [1, 0]
    elif country == "Spain":
        countries = [0, 1]

    # Encode gender
    gender = row['Gender']
    if gender == 'Male':
        encoded_gender = 1
    else:
        encoded_gender = 0

    # Make Prediction
    prediction = model.predict(sc.transform(np.array([[countries[0], countries[1], row['CreditScore'], encoded_gender, row['Age'], row['Tenure'], row['Balance'], row['NumOfProducts'], row['HasCrCard'], row['IsActiveMember'], row['EstimatedSalary']]])))

    # Process the prediction result for human
    if prediction > 0.5:
        return "The customer will leave the bank"
    else:
        return "The customer won't leave the bank"
    

# @app.route('/download_prediction/<filename>', methods=['GET'])
# def download_prediction(filename):
# # Logic to handle downloading the predicted results file
# # For demonstration purposes, assuming the file is stored in the same directory
#     return send_file(filename, as_attachment=True)

# @app.route('/download_prediction', methods=['GET'])
# def download_prediction():
#    # Load the Excel file into a DataFrame
#     df = pd.read_excel('predicted_results.xlsx')  # Load the previously generated Excel file
    
#     # Create a BytesIO object to hold the file contents
#     output = BytesIO()
    
#     # Write the DataFrame to the BytesIO object
#     df.to_excel(output, index=False)
    
#     # Set the file pointer to the beginning of the BytesIO object
#     output.seek(0)
    
#     # Create a Flask response with the file attachment
#     response = make_response(send_file(output, as_attachment=True, attachment_filename='predicted_results.xlsx'))
    
    
#     # Set the appropriate content type for the response
#     # response.headers['Content-Type'] = 'application/vnd.ms-excel'
#     response.headers['Content-Type'] = 'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet'
    
#     return response

        

if __name__ == '__main__':
    app.run()
