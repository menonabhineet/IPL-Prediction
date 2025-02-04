import numpy as np
from flask import Flask, request, jsonify, render_template
import pickle

app = Flask(__name__)
model = pickle.load(open('xgb.pkl', 'rb'))

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict',methods=['POST'])
def predict():

    int_features = [int(x) for x in request.form.values()]
    final_features = [np.array(int_features)]
    prediction = model.predict(final_features)

    output = round(prediction[0], 2)
    if output == '0':
        op1=int_features[1]
    else:
        op1=int_features[0]
    
    if op1==0:
        op='MI'
    elif op1==1:
        op='KKR'
    elif op1==2:
        op='RCB'
    elif op1==3:
        op='CSK'
    elif op1==4:
        op='RR'
    elif op1==5:
        op='DC'
    elif op1==6:
        op='GL'
    elif op1==7:
        op='PBKS'
    elif op1==8:
        op='SRH'
    elif op1==9:
        op='RPS'

    return render_template('index.html', prediction_text='Winner should be  {}'.format(op))

@app.route('/results',methods=['POST'])
def results():

    data = request.get_json(force=True)
    prediction = model.predict([np.array(list(data.values()))])

    output = prediction[0]
    return jsonify(output)

if __name__ == "__main__":
    app.run(debug=True)