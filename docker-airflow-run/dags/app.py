#from flask import Flask, render_template, request, redirect, url_for, flash, jsonify
from flask import Flask, jsonify, request
import pandas as pd
import json
import os
import pickle
import datetime


from src import config
from src import inference
from src import helpers
from src import preprocess


app = Flask(__name__)

@app.route("/ping", methods=["GET"])
def ping():
    return jsonify({"status": "ok", "datetime": str(datetime.datetime.now())})

@app.route('/predict', methods=['POST'])
def predict():
    X = request.json.get('X')
    if X is None:
        return jsonify({'error': 'No input data provided'}), 400
    if not isinstance(X, list):
        return jsonify({'error': 'Input data must be a list of key-value items'}), 400
    
    print("Received data:", X)
    X = pd.DataFrame(X)
    print("DataFrame:", X)

    if "customer_id" not in X.columns:
        return jsonify({'error': 'Input data must contain a customer_id column'}), 400

    ref_job_id = helpers.get_latest_deployed_job_id()
    if ref_job_id is None:
        return jsonify({'error': 'No deployed models available'}), 500

    X = preprocess.preprocess_data(X, mode="inference", rescale=False, ref_job_id=ref_job_id)
    print("Preprocessed Data:", X)

    r = inference.make_predictions(X, predictors=config.PREDICTORS)
    print("Predictions:", r)
    
    return jsonify(r)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5100, debug=True)
