from flask import Flask, jsonify, request
import joblib
from helper import *

import flask
    
app = Flask(__name__)

@app.route('/')
@app.route('/index')
def index():
    return flask.render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    clf = joblib.load('model/model.pkl')
    count_vect = joblib.load('model/count_vect.pkl')
    to_predict_list = request.form.to_dict()
    original_comment = to_predict_list['comment']
    comment = clean_text(original_comment)
    pred = clf.predict(count_vect.transform([comment]))
    if pred[0]:
        prediction = "Insincere"
    else:
        prediction = "Sincere"

    return flask.render_template('predict.html', prediction=prediction, comment=original_comment)

if __name__ == "__main__":
    app.run(host='0.0.0.0')