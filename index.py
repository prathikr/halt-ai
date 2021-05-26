import numpy as np
import pandas as pd
import pickle
from math import sqrt
from flask import Flask, render_template, request
app = Flask(__name__)

@app.route('/')
def index():
   return render_template('index.html')

@app.route('/prediction', methods=['POST'])
def prediction():
   data = request.form  # data is empty
   cleaned_data = {}
   for key in list(data.keys()):
      if data[key] == 'NaN':
         cleaned_data[key] = [np.NAN]
      else:
         cleaned_data[key] = [int(data[key])]
   
   X = pd.DataFrame(cleaned_data)

   preds = np.array([])
   for i in range(5):
      loaded_model = pickle.load(open('./models/model' + str(i+1) + '.sav', 'rb'))
      pred = loaded_model.predict_proba(X)
      preds = np.append(preds, pred[0][0])

   mean = round(preds.mean(), 4)

   confidence_level = 2.776 # corresponding z-score value for 95% confidence and 4 degrees of freedom (since we did 5 fold cv)
   ci = round(confidence_level * preds.std() / sqrt(len(preds)), 4)
   result = str(mean) + " +/- " + str(ci)

   return render_template('prediction.html', value=result)


if __name__ == '__main__':
   app.run(debug=False, host='0.0.0.0')
