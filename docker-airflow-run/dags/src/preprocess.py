import traceback
import pandas as pd
import numpy as np
import re
import os
import json
import pickle
import datetime

from sklearn.preprocessing import MinMaxScaler, StandardScaler


from functools import reduce
try:
    from src import config
    from src import helpers
except ImportError:
    import config
    import helpers

"""
Improvements:
    1. Use LabelEncoder to encode categorical variables.
    2. Use LabelBinarizer to encode categorical variables.
    3. Use OrdinalEncoder to encode categorical variables.


"""
cat_vars = list(map(str.lower, config.CAT_VARS))
num_vars = list(map(str.lower, config.NUM_VARS))
date_vars = list(map(str.lower, config.DATETIME_VARS))
exc_vars = list(map(str.lower, config.EXC_VARIABLES))
engineered_vars = {
    "categorical": ["application_year", "application_month", "application_week", "application_day", "application_season"],
    "numerical": ["current_credit_balance_ratio"],
    "date": ["application_date"]
}

######  missing values ######
def get_variables_with_missing_values(df:pd.DataFrame) -> pd.DataFrame:
    """
    Get variables with missing values.
    :param df: DataFrame
    :return: DataFrame
    """
    missing_counts = df.isnull().sum()
    return missing_counts[missing_counts>0].index.tolist()

def impute_missing_values(df:pd.DataFrame, method:str="basic", mode:str=None, cat_vars:list=config.CAT_VARS, num_vars:list=config.NUM_VARS, job_id:str="") -> pd.DataFrame:
    """
    Treat missing values.
    
    :param df: DataFrame
    :param method: str, "basic" or "advanced"
        For basic method
            If the column with missing values is a categorical variable, we can impute it with the most frequent value.
            If the column with missing values is a numerical variable, we can impute it with the mean value.
        For advanced method
    :param mode: str, "training" or "inference"
    :return: DataFrame
    """
    assert mode in ("training", "inference"), f"mode must be either 'training' or 'inference', but got {mode}"
    assert method in ["basic", "advanced"], f"{method} is not a valid methods (basic, advanced)"
    if mode=="training":
        model = {
            "method": method,
            "imputes": dict()
        }
        for col in df.columns:
            print("[INFO] Treating missing values in column:", col)
            model["imputes"][col] = dict()
            if method=="basic":
                if col in set(cat_vars):
                    model["imputes"][col]['mode'] = df[df[col].notnull()][col].mode()[0]
                elif col in set(num_vars):
                    model["imputes"][col]['mean'] = df[df[col].notnull()][col].mean()
                elif col in set(config.DATETIME_VARS):
                    model["imputes"][col]['mode'] = df[df[col].notnull()][col].mode()[0]
                elif col in ["customer_id", "churn"]:
                    pass
                else:
                    raise ValueError(f"[ERROR]{col} is not a valid variable")
            if method=="advanced":
                raise(NotImplementedError)
        helpers.save_model_as_pickle(model, f"{job_id}_missing_values_model")
        return impute_missing_values(df, method=method, mode="inference", cat_vars=cat_vars, num_vars=num_vars, job_id=job_id)
    else:
        model = helpers.load_model_from_pickle(model_name=f"{job_id}_missing_values_model")
        cols = get_variables_with_missing_values(df)
        method = model["method"]
        if method=="basic":
            for col in cols:
                if col in set(cat_vars):
                    df[col].fillna(model["imputes"][col]['mode'], inplace=True)
                elif col in set(num_vars):
                    df[col].fillna(model["imputes"][col]['mean'], inplace=True)
                elif col in set(config.DATETIME_VARS):
                    df[col].fillna(model["imputes"][col]['mode'], inplace=True)
                elif col in ["customer_id", "churn"]:
                    pass
                else:
                    raise ValueError(f"[ERROR]{col} is not a valid variable. Pre-trained vairables: {list(model['imputes'].keys())}")
        if method=="advanced":
            raise(NotImplementedError)
    return df

###### enforcing datatypes ######
def enforce_numeric_to_float(x: str) -> float:
    """
    Convert numeric to float. To ensure that all stringified numbers are converted to float.
    :param x: str
    :return: float
    """
    try:
        return float(re.sub("[^0-9.]","", str(x)))
    except ValueError:
        return np.nan

def enforce_datatypes_on_variables(df:pd.DataFrame, cat_vars:list=[], num_vars:list=[]) -> pd.DataFrame:
    """
    Transform variables.
    :param df: DataFrame
    :return: DataFrame
    """
    #df["application_time"] = pd.to_datetime(df["application_time"])
    for var in num_vars:
        df[var] = df[var].apply(lambda x: enforce_numeric_to_float(x))
    for var in cat_vars:
        df[var] = df[var].astype(str)
    return df


###### encoding categorical variables ######
def country_to_int(x: str) -> int:
    """
    Convert country status to int.
    :param x: str, lower cased country
    :return: int
    """

    assert x in ("france", "spain", "germany") or isinstance(x, int), f"{x} is not a valid country status and is not an integer"
    if x.strip() =="france":
        return 1
    if x.strip() =="spain":
        return 2
    if x.strip() =="germany":
        return 3
    return x


def gender_to_int(x: str) -> int:
    """
    Convert gender status to int.
    :param x: str, lower cased gender status
    :return: int
    """

    assert x in ("male", "female") or isinstance(x, int), f"{x} is not a valid loan status and is not an integer"
    if x.strip()=="male":
        return 1
    if x.strip()=="female":
        return 2
    return x


 
def credit_card_to_int(x: str) -> int:
    """
    Converts a string that represents a number into an integer.

    :param x: String to convert
    :return: Integer representation of the string
    :raises ValueError: If the string does not represent a valid integer
    """
    try:
        # Attempt to convert the string to an integer
        return int(x)
    except ValueError as e:
        # If conversion fails, raise an error with a custom message
        raise ValueError(f"Cannot convert '{x}' to int: {e}")

def active_member_to_int(x: str) -> int:
    """
    Converts a string that represents a number into an integer.

    :param x: String to convert
    :return: Integer representation of the string
    :raises ValueError: If the string does not represent a valid integer
    """
    try:
        # Attempt to convert the string to an integer
        return int(x)
    except ValueError as e:
        # If conversion fails, raise an error with a custom message
        raise ValueError(f"Cannot convert '{x}' to int: {e}")

def products_number_to_int(x: str) -> int:
    """
    Converts a string that represents a number into an integer.

    :param x: String to convert
    :return: Integer representation of the string
    :raises ValueError: If the string does not represent a valid integer
    """
    try:
        # Attempt to convert the string to an integer
        return int(x)
    except ValueError as e:
        # If conversion fails, raise an error with a custom message
        raise ValueError(f"Cannot convert '{x}' to int: {e}")




def encode_categorical_variables(df:pd.DataFrame, mode="training", job_id:str="") -> pd.DataFrame:
    """
    Encode categorical variables.
    :param df: DataFrame
    :param purpose_encode_method: str, choose from "ranking", "weighted ranking", "relative ranking"
    :return: DataFrame
    """
    assert mode in ("training", "inference"), f"{mode} is not a valid mode (training , inference)"
    assert isinstance(job_id, str)
    for col in config.CAT_VARS:
        assert col in df.columns, f"{col} not in {df.columns}"
        df[col] = df[col].str.lower()

    df["country"] = df["country"].apply(lambda x: country_to_int(x))
    df["gender"] = df["gender"].apply(lambda x: gender_to_int(x)) 
    df['products_number'] =  df['products_number'].apply(lambda x: products_number_to_int(x))
    df['active_member'] =  df['active_member'].apply(lambda x: active_member_to_int(x)) 
    df['credit_card'] =  df['credit_card'].apply(lambda x: credit_card_to_int(x))  
    return df

###### engineer new variables ######
def month_to_season(month:int) -> int:
    """
    Convert date to season.
    :param m: int, month between 1 and 12
    :return: int
    """
    if month in [1, 2, 3]:
        return 1
    elif month in [4, 5, 6]:
        return 2
    elif month in [7, 8, 9]:
        return 3
    elif month in [10, 11, 12]:
        return 4
    else:
        return np.nan

def engineer_variables(df:pd.DataFrame) -> pd.DataFrame:
    """
    Engineer variables.
    :param df: DataFrame
    :return: DataFrame
    """
    for col in ["application_time"]:
        assert col in df.columns, f"{col} not in {df.columns}"

    df["application_date"] = df["application_time"].dt.date
    df["application_year"] = df["application_time"].dt.year
    df["application_month"] = df["application_time"].dt.month
    df["application_week"] = df["application_time"].dt.week
    df["application_day"] = df["application_time"].dt.day
    df["application_season"] = df["application_month"].apply(lambda x: month_to_season(x))
    df["current_credit_balance_ratio"] = (df["current_credit_balance"]/df["current_loan_amount"]).fillna(0.0)
    return df

def split_train_test(df:pd.DataFrame, test_size:float, method:str='time based'):
    """
    Split data into train and test.
    :param df: DataFrame
    :param test_size: float, between 0 and 1
    :param method: str, 'time based' or 'random'
    :return: (DataFrame, DataFrame)
    """
    if method=='random':
        return df.sample(frac=1, random_state=config.RANDOM_STATE).iloc[:int(len(df)*test_size)], df.sample(frac=1, random_state=config.RANDOM_STATE).iloc[int(len(df)*test_size):]
    if method=='time based':
        unique_dates = sorted(df["application_date"].unique())
        
        train_dates = unique_dates[:int(len(unique_dates)*(1-test_size))]
        test_dates = unique_dates[unique_dates.index(train_dates[-1])+1:]
        train_df = df[df["application_date"].isin(train_dates)]
        test_df = df[df["application_date"].isin(test_dates)]

        return train_df, test_df
    raise(ValueError(f"{method} is not a valid method (time based, random)"))

def rescale_data(df:pd.DataFrame, method:str='standardize', mode:str='training', columns:list=[], job_id:str="") -> pd.DataFrame:
    """
    Rescale data.
    :param df: DataFrame
    :param method: str, 'standardize' or 'minmax'
    :param mode: str, 'training' or 'inference'
    :return: DataFrame
    """
    assert method in ('standardize', 'minmax'), f"{method} is not a valid method (standardize, minmax)"
    assert mode in ('training', 'inference'), f"{mode} is not a valid mode (training, inference)"
    for col in columns:
        assert col in df.columns

    if mode=='training':
        if method=='standardize':
            scaler = StandardScaler()
            scaler.fit(df[columns])
        if method=='minmax':
            scaler = MinMaxScaler()
            scaler.fit(df[columns])
        model = {
            'scaler': scaler,
            'method': method,
        }

        helpers.save_model_as_pickle(model, f"{config.PATH_DIR_MODELS}/{job_id}_numerical_scaler.pkl")
        df[list(map(lambda x: f"{method}_{x}", columns))] = scaler.transform(df[columns])
        return df
    if mode=='inference':
        model = helpers.load_model_from_pickle(model_name=f"{job_id}_numerical_scaler.pkl")
        scaler = model['scaler']
        method = model['method']
        for col in columns:
            try:
                df[col].astype(float)
            except:
                print("[DEBUG] Column skipped:", col)
        df[list(map(lambda x: f"{method}_{x}", columns))] = scaler.transform(df[columns])
        return df

def preprocess_data(df:pd.DataFrame, mode:str, job_id:str=None, rescale=False, ref_job_id:str=None) -> pd.DataFrame:
    """
    Pre-process data and save preprocessed datasets for later use.
    :param df: DataFrame
    :param mode: str, 'training' or 'inference'
    :param job_id: str, job_id for the preprocessed dataset
    :param rescale: bool, whether to rescale data.
    :param ref_job_id: str, job_id of the last deployed model. Usefull when doing inference.
    :return: DataFrame
    """
    assert mode in ('training', 'inference')
    
    if mode=='training':
        assert config.TARGET in df.columns, f"{config.TARGET} not in {df.columns}"

    df.columns = list(map(str.lower, df.columns))
    initial_size = df.shape[0]
    if mode=='training':
        df = df[df["customer_id"].notnull() & df["churn"].notnull()]
    #if mode=='training':
    #    df["loan_status"] = df["loan_status"].str.lower()
    if df.shape[0] != initial_size:
        print(f"[WARNING] Dropped {initial_size - df.shape[0]} rows with null values in (customer_id, churn)")
    df = enforce_datatypes_on_variables(df, cat_vars=config.CAT_VARS, num_vars=config.NUM_VARS)
    #df = engineer_variables(df)
    if mode=='training':
        # split train and test data before encoding categorical variables and imputing missing values
        train_df, test_df = split_train_test(df, config.TEST_SPLIT_SIZE, method=config.SPLIT_METHOD)
        train_df = encode_categorical_variables(train_df, mode="training", job_id=job_id)
        train_df = impute_missing_values(train_df, method="basic", mode="training", job_id=job_id)
        if rescale:
            train_df = rescale_data(train_df, method=config.RESCALE_METHOD, mode="training", columns=num_vars)# + engineered_vars["numerical"])
        helpers.save_dataset(train_df, os.path.join(config.PATH_DIR_DATA, "preprocessed", f"{job_id}_training.csv"))
        preprocess_data(test_df, mode="inference", job_id=job_id, ref_job_id=job_id)
    else:
        # if mode is infer, no need to split train and test data
        test_df = encode_categorical_variables(df, mode="inference", job_id=ref_job_id)
        test_df = impute_missing_values(test_df, method="basic", mode="inference", job_id=ref_job_id)
        if rescale:
            test_df = rescale_data(test_df, method=config.RESCALE_METHOD, mode="inference", columns=num_vars)# + engineered_vars["numerical"])
        helpers.save_dataset(test_df, os.path.join(config.PATH_DIR_DATA, "preprocessed", f"{job_id}_inference.csv"))
    return test_df


if __name__=='__main__':
    preprocess_data(df=helpers.load_dataset(os.path.join(config.PATH_DIR_DATA, "raw", "loan eligibility data", "LoansTraining.csv")), mode="training")
