RUN_LOCAL = False
if RUN_LOCAL:
    PATH_DIR_DATA = "../dags/data"
    PATH_DIR_MODELS = "../dags/models"
    PATH_DIR_RESULTS = "../dags/results"
    PATH_TO_CREDENTIALS = "../dags/Creds.json"
    PATH_TO_APP_SHELL = "../dags/app.sh"
else:
    PATH_DIR_DATA = "/opt/airflow/dags/data"
    PATH_DIR_MODELS = "/opt/airflow/dags/models"
    PATH_DIR_RESULTS = "/opt/airflow/dags/results"
    PATH_TO_CREDENTIALS = "/opt/airflow/dags/Creds.json"
    PATH_TO_APP_SHELL = "/opt/airflow/dags/app.sh"

RANDOM_SEED = 42
TEST_SPLIT_SIZE = 0.3
PROB_THRESHOLD = 0.5
SPLIT_METHOD = "random"
RANDOM_STATE = 101

# lowest acceptable difference between the performances of the same model on two different datasets
MODEL_DEGRADATION_THRESHOLD = 0.2
ASSOCIATION_DEGRADATION_THRESHOLD = 0.3

# lowest acceptable performance of either accuracy, precision, recall, f1 or auc depending on the classification usecase
MODEL_PERFORMANCE_THRESHOLD = 0.7 
MODEL_PERFORMANCE_METRIC = "auc"

IDENTIFIERS = ['customer_id']
TARGET = 'churn'
DATETIME_VARS = ['application_time']
EXC_VARIABLES = [
    'application_time'
    ]

PURPOSE_ENCODING_METHOD = "weighted ranking" # choose from (ranking, one-hot, weighted ranking, relative ranking)
RESCALE_METHOD = "standardize" # choose from (standardize, minmax, None)
CAT_VARS = [
    'country',
    'products_number', 
    'gender', 
    'credit_card',
    'active_member',
    ]



NUM_VARS = [
    'credit_score',
    'age', 
    'tenure', 
    'balance', 
    'estimated_salary', 
    ]

PREDICTORS = [
    'country',
    'products_number', 
    'gender', 
    'credit_card',
    'active_member',
    'credit_score', 
    'age', 
    'tenure', 
    'balance', 
    'estimated_salary', 
]

STAGES = [
    "etl", "preprocess", "training", "testing", "inference", "postprocess", "preprocess-training", "preprocess-inference", "report", "driftcheck",
    "etl_report", "raw_data_drift-report", "deploy"
    ]
STATUS = ["pass", "fail", "skipped", "started"]
JOB_TYPES = ["training", "inference", None]


