from airflow import DAG
from airflow.providers.slack.operators.slack import SlackAPIPostOperator
from airflow.operators.python import PythonOperator, BranchPythonOperator
from airflow.operators.dummy import DummyOperator
from datetime import datetime, timedelta

from src import config, helpers, etl, preprocess, train, inference

dag_id = "churn_pipeline_monitoring"

def evaluate_churn_scores(ti, **kwargs):
    """
    Evaluate churn scores and determine if more than 10 customers have scores > 90%.
    """
    churn_scores = ti.xcom_pull(task_ids='predict_churn_scores', key='return_value')
    high_risk_customers = [score for score in churn_scores if score['churn_probability'] > 0.9]

    if len(high_risk_customers) > 10:
        message = f"Alert: More than 10 customers have a churn score above 90%. Count: {len(high_risk_customers)}"
        ti.xcom_push(key='alert_message', value=message)
        return 'send_slack_alert'
    return 'no_action_required'

def create_dag(dag_id):
    """
    Create a DAG to process and monitor churn scores.
    """
    with DAG(
        dag_id=dag_id,
        schedule_interval='@daily',
        default_args={
            'owner': 'airflow',
            'start_date': datetime(2024, 1, 1),
            'retries': 1,
            'retry_delay': timedelta(minutes=5),
        },
        catchup=False
    ) as dag:

        start = DummyOperator(task_id='start')

        predict_churn_scores = PythonOperator(
            task_id='predict_churn_scores',
            python_callable=inference.make_predictions,
            op_kwargs={'model_id': 'latest_model'}
        )

        evaluate_scores = BranchPythonOperator(
            task_id='evaluate_churn_scores',
            python_callable=evaluate_churn_scores
        )

        send_slack_alert = SlackAPIPostOperator(
            task_id='send_slack_alert',
            channel='#alerts',
            text="{{ ti.xcom_pull(task_ids='evaluate_churn_scores', key='alert_message') }}",
            slack_conn_id='slack',  # Make sure the Slack connection ID is set correctly in the Airflow UI
            username='Airflow'
        )

        no_action_required = DummyOperator(task_id='no_action_required')

        end = DummyOperator(task_id='end')

        start >> predict_churn_scores >> evaluate_scores >> [send_slack_alert, no_action_required] >> end

        return dag

globals()[dag_id] = create_dag(dag_id)
