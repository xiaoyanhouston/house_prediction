import numpy as np
from flask import Flask, request,render_template
import pickle
from datetime import date
import pandas as pd
# Get today's date
today = date.today()

app = Flask(__name__)

#file_path= 'C:/Users/xiaoy/Desktop/DataGalacier/github_repo/house_prediction/'
#model = pickle.load(open('C:/Users/xiaoy/Desktop/DataGalacier/github_repo/house_prediction/best_model.pkl', 'rb'))
#model= joblib.load('C:/Users/xiaoy/Desktop/DataGalacier/github_repo/house_prediction/model_best.pkl')
model= pickle.load(open('best_model.pkl','rb'))

@app.route('/')
def home():
    return render_template('index.html')

# Get the first day of the current month
first_day_of_month = today.replace(day=1)
def get_hpi(input_date=first_day_of_month):
    dallas_hpif= pd.read_csv("C:/Users/xiaoy/Desktop/DataGalacier/github_repo/house_prediction/Dallas_HPIF.csv")
    dallas_hpif['DATE'] = pd.to_datetime(dallas_hpif['DATE'], format='%m/%d/%Y').dt.date
    dallas_hpif=dallas_hpif.sort_values(by='DATE')
    dallas_hpif['hpi_1month_lag'] = dallas_hpif['HPI'].shift(1)  # Lag by 1 month
    dallas_hpif['hpi_3month_lag'] = dallas_hpif['HPI'].shift(3)  # Lag by 3 month
    dallas_hpif['hpi_6month_lag'] = dallas_hpif['HPI'].shift(6)  # Lag by 6 periods
    dallas_hpif['hpi_1m_pct'] = (dallas_hpif['HPI']/dallas_hpif['hpi_1month_lag']-1)
    dallas_hpif['hpi_3m_pct'] = (dallas_hpif['HPI']/dallas_hpif['hpi_3month_lag']-1)
    dallas_hpif['hpi_6m_pct'] = (dallas_hpif['HPI']/dallas_hpif['hpi_6month_lag']-1)
    dallas_hpif['hpi_1m_pct_lag'] = dallas_hpif['hpi_1m_pct'].shift(1)
    dallas_hpif['hpi_3m_pct_lag'] = dallas_hpif['hpi_3m_pct'].shift(1)
    dallas_hpif['hpi_6m_pct_lag'] = dallas_hpif['hpi_6m_pct'].shift(1)
    # Calculate the moving average
    dallas_hpif['hpi_3m_ma'] = dallas_hpif['HPI'].rolling(window=3).mean()

    dallas_hpif['hpi_6m_ma'] = dallas_hpif['HPI'].rolling(window=6).mean()
    dallas_hpif['hpi_3m_ma_lag'] = dallas_hpif['hpi_3m_ma'].shift(1)
    dallas_hpif['hpi_6m_ma_lag'] = dallas_hpif['hpi_6m_ma'].shift(1)
    keep_list=["hpi_1month_lag",
    "hpi_3month_lag",
    "hpi_6month_lag",
    "hpi_1m_pct",
    "hpi_3m_pct",
    "hpi_6m_pct",
    "hpi_1m_pct_lag",
    "hpi_3m_pct_lag",
    "hpi_6m_pct_lag",
    "hpi_3m_ma",
    "hpi_6m_ma",
    "hpi_3m_ma_lag",
    "hpi_6m_ma_lag"
    ]
    hpi_row= dallas_hpif[dallas_hpif['DATE']== input_date][keep_list]
    row_list = hpi_row.values[0].tolist()
    return row_list
    

@app.route('/predict',methods=['POST'])
def predict():
    '''
    For rendering results on HTML GUI,
    variable:
        BEDS
        BATHS
        SQUAREFEET
        LOTSIZE
        YEARBUILT
        hpi_1month_lag
        hpi_3month_lag
        hpi_6month_lag
        hpi_1m_pct
        hpi_3m_pct
        hpi_6m_pct
        hpi_1m_pct_lag
        hpi_3m_pct_lag
        hpi_6m_pct_lag
        hpi_3m_ma
        hpi_6m_ma
        hpi_3m_ma_lag
        hpi_6m_ma_lag
        total_beds_baths
        sqft_pbb
        age

    '''
    input_value=[int(x) for x in request.form.values()]
    hpi=get_hpi()
    rest=[input_value[0]+input_value[1],input_value[2]/(input_value[0]+input_value[1]),today.year- input_value[4]]
    int_features = input_value+hpi+rest
    final_features = [np.array(int_features)]
    prediction = model.predict(final_features)

    output = round(prediction[0], 2)

    return render_template('index.html', prediction_text='House price should be $ {}'.format(output))

if __name__ == "__main__":
    app.run(debug=True)