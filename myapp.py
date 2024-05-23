import numpy as np
from flask import Flask, request,render_template
import pickle
import joblib

myapp = Flask(__name__)

file_path= 'C:/Users/xiaoy/Desktop/DataGalacier/github_repo/house_prediction/'
#model = pickle.load(open('C:/Users/xiaoy/Desktop/DataGalacier/github_repo/house_prediction/best_model.pkl', 'rb'))
# model= joblib.load('C:/Users/xiaoy/Desktop/DataGalacier/github_repo/house_prediction/model_best.pkl')
model= pickle.load(open('model.pkl','rb'))

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict',methods=['POST'])
def predict():
    '''
    For rendering results on HTML GUI
    '''
    int_features = [int(x) for x in request.form.values()]
    final_features = [np.array(int_features)]
    prediction = model.predict(final_features)

    output = round(prediction[0], 2)

    return render_template('index.html', prediction_text='House price should be $ {}'.format(output))

if __name__ == "__main__":
    myapp.run(debug=True)