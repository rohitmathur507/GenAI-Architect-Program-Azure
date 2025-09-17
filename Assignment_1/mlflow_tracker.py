import json

import mlflow

from config import MLFLOW_TRACKING_URI

mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)

# Create or set the experiment for sentiment analysis
EXPERIMENT_NAME = "Rohit_Mathur_Market_Sentiment_Analysis"
try:
    experiment_id = mlflow.create_experiment(EXPERIMENT_NAME)
    print(f"Created new experiment: {EXPERIMENT_NAME} with ID: {experiment_id}")
except mlflow.exceptions.MlflowException:
    # Experiment already exists
    experiment_id = mlflow.get_experiment_by_name(EXPERIMENT_NAME).experiment_id
    print(f"Using existing experiment: {EXPERIMENT_NAME} with ID: {experiment_id}")

mlflow.set_experiment(EXPERIMENT_NAME)


def log_sentiment_analysis(
    company_name: str, stock_code: str, news_data: str, result: dict
):
    # Log within the existing active run instead of starting a new one
    mlflow.log_param("company_name", company_name)
    mlflow.log_param("stock_code", stock_code)
    mlflow.log_metric("confidence_score", result.get("confidence_score", 0.0))
    mlflow.log_text(news_data, "input_news.txt")
    mlflow.log_text(json.dumps(result, indent=2), "sentiment_output.json")
    mlflow.set_tag("sentiment", result.get("sentiment", "Unknown"))
    mlflow.set_tag("model_type", "sentiment_analyzer")


def start_mlflow_run():
    # End any existing run before starting a new one
    if mlflow.active_run():
        mlflow.end_run()
    return mlflow.start_run()


def end_mlflow_run():
    if mlflow.active_run():
        mlflow.end_run()


def ensure_no_active_run():
    """Utility function to ensure no active MLflow run exists."""
    if mlflow.active_run():
        mlflow.end_run()


def get_experiment_info():
    """Get information about the current experiment."""
    experiment = mlflow.get_experiment_by_name(EXPERIMENT_NAME)
    return {
        "experiment_id": experiment.experiment_id,
        "experiment_name": experiment.name,
        "artifact_location": experiment.artifact_location,
        "lifecycle_stage": experiment.lifecycle_stage,
    }
