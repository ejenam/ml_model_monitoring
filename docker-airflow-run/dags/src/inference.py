import pandas as pd
import numpy as np
import datetime
import os
import logging
import json
import pickle
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, BaseEnsemble
try:
    from src import config
    from src import helpers
    from src import preprocess
    from src import etl
except:
    import config
    import helpers
    import preprocess
    import etl

def load_model():
    if not os.path.isfile(os.path.join(config.PATH_DIR_MODELS, "deploy_report.json")):
        json.dump({"prediction_model": None}, open(os.path.join(config.PATH_DIR_MODELS, "deploy_report.json"), "w"))
    filename = json.load(open(os.path.join(config.PATH_DIR_MODELS, "deploy_report.json"), "r"))["prediction_model"]
    with open(os.path.join(config.PATH_DIR_MODELS, filename), "rb") as f:
        model = pickle.load(f)
    return model
model = load_model()

def batch_inference(job_id:str, ref_job_id:str, start_date:datetime.date, end_date:datetime.date=datetime.date.today(), predictors=[]) -> dict:
    """
    :param job_id: str
    :param ref_job_id: str
    :param start_date: datetime.date
    :param end_date: datetime.date
    :param predictors: list
    :return: dict
    """
    model_type = helpers.get_model_type(ref_job_id)
    model = helpers.load_model_from_pickle(f"{ref_job_id}_{model_type}")
    etl.collect_data(start_date=start_date, end_date=end_date, job_id=job_id)
    df = helpers.load_dataset(helpers.locate_raw_data_filename(job_id))
    preprocess.preprocess_data(df, mode="inference", job_id=job_id, ref_job_id=ref_job_id, rescale=False, job_date=start_date, inner_call=False)
    _, test_filename = helpers.locate_preprocessed_filenames(job_id)
    df = helpers.load_dataset(test_filename)
    df['churn_probability'] = model.predict_proba(df[predictors])
    high_risk_customers = df[df['churn_probability'] > 0.9]
    return dict(high_risk_customers[['customer_id', 'prediction']].values)

    
def make_predictions(df: pd.DataFrame, model: BaseEnsemble = None, predictors=[]) -> pd.DataFrame:
    """
    Make predictions using the provided model and return churn probabilities along with high-risk assessment.
    
    :param df: pd.DataFrame containing the input features for prediction.
    :param model: Trained model used for making predictions. If None, a default model is loaded.
    :param predictors: List of feature names used by the model for predictions.
    :return: pd.DataFrame containing customer IDs, churn probabilities, and high-risk flag.
    """
    if model is None:
        model = load_model()  # Assumes a function that loads your default model
    if model == None:
        return {"error": "No model deployed"}

    # Predict churn probabilities
    df['churn_probability'] = model.predict_proba(df[predictors])[:, 1]
    # Determine if each customer is considered high-risk based on the threshold
    df['high_risk'] = (df['churn_probability'] > 0.9).astype(int)  # 1 for high-risk, 0 for not high-risk

    # Prepare the result as a list of dictionaries
    result = df[['customer_id', 'churn_probability', 'high_risk']].to_dict(orient='records')
    return result






if __name__=='__main__':
    predictors = config.PREDICTORS
    job_date = datetime.date(2016, 11, 1)
    job_id = helpers.generate_uuid()
    ref_job_id = helpers.get_latest_training_job_id(status="pass")
    preds = batch_inference(job_id, ref_job_id, job_date, predictors=predictors)
    print(preds)

    
    