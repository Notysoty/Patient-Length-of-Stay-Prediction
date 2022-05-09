# importing necessary libraries and functions
import pandas as pd
from flask import Flask, request, render_template
import pickle

app = Flask(__name__) #Initialize the flask App
patientid_dict = pickle.load(open('patientid_dict.pkl', 'rb'))
No_Hospitals_in_city_dict = pickle.load(open('No_Hospitals_in_city_dict.pkl', 'rb'))
No_City_in_region_dict = pickle.load(open('No_City_in_region_dict.pkl', 'rb'))
no_of_wards_dict = pickle.load(open('no_of_wards_dict.pkl', 'rb'))
standardscalar = pickle.load(open('standardscalar.pkl', 'rb'))
pipe = pickle.load(open('pipe.pkl', 'rb'))
model = pickle.load(open('tuned_model.pkl', 'rb')) # loading the trained model
get_dummy = pickle.load(open('X_train_columns.pkl', 'rb'))
@app.route('/') # Homepage
def home():
    return render_template('index.html')

@app.route('/predict',methods=['POST'])
def predict():
    '''
    For rendering results on HTML GUI
    '''
    
    # retrieving values from form
    init_features = [[x for x in request.form.values()]]
    df = pd.DataFrame(init_features, columns =['Hospital_code', 'patientid', 'Hospital_type_code', 'City_Code_Hospital', 'Hospital_region_code',
       'Available Extra Rooms in Hospital', 'Department', 'Ward_Type',
       'Ward_Facility_Code', 'Bed Grade', 'City_Code_Patient',
       'Type of Admission', 'Severity of Illness', 'Visitors with Patient',
       'Age', 'Admission_Deposit'] )
    df['No of times patient appears in data'] = df.patientid.map(patientid_dict)
    df['No_Hospitals_in_city'] = df['City_Code_Hospital'].map(No_Hospitals_in_city_dict)
    df['No_City_in_region'] = df['Hospital_region_code'].map(No_City_in_region_dict)
    df['no_of_wards'] = df['Hospital_code'].map(no_of_wards_dict)
    df.drop(labels = ['patientid', 'Hospital_code'], axis = 1, inplace = True)
    df = pd.get_dummies(df)
    df = get_dummy.mask(df != 0, other = df)
    df = pd.DataFrame(data = standardscalar.transform(df), columns = df.columns)
    df = pipe.transform(df)
    prediction = model.predict(df) # making prediction
    prediction = pd.Series(prediction.flatten()).map({0:'<20',1:'21-30',2:'31-40', 3:'>40' })
    return render_template('index.html', prediction_text='Predicted Stay: {}'.format(prediction)) # rendering the predicted result

if __name__ == "__main__":
    app.run(debug=True, use_reloader=False)