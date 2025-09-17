import mlflow
from langchain_core.runnables import RunnableLambda, RunnablePassthrough
from operator import itemgetter

from mlflow_tracker import log_sentiment_analysis
from stock_symbol import stock_symbol_extractor
from news_fetcher import news_fetcher
from sentiment_analyzer import sentiment_analyzer


def log_and_return(inputs):
    """Log results to MLflow and return as dictionary."""
    company_name = inputs["company_name"]
    stock_code = inputs["stock_code"]
    news_data = inputs["news_data"]
    result = inputs["result"]

    # Log to MLflow within the existing run
    log_sentiment_analysis(company_name, stock_code, news_data, result.dict())
    return result.dict()


# LCEL Chain using modular components
market_sentiment_chain = (
    RunnablePassthrough.assign(
        stock_code=itemgetter("company_name") | stock_symbol_extractor
    )
    | RunnablePassthrough.assign(news_data=itemgetter("stock_code") | news_fetcher)
    | RunnablePassthrough.assign(
        result=RunnableLambda(
            lambda x: sentiment_analyzer.invoke(
                {
                    "company_name": x["company_name"],
                    "stock_code": x["stock_code"],
                    "news_data": x["news_data"],
                }
            )
        )
    )
    | RunnableLambda(log_and_return)
)


def analyze_company_sentiment(company_name: str) -> dict:
    """
    Main function to analyze company sentiment using LCEL chain.

    Args:
        company_name (str): Name of the company to analyze

    Returns:
        dict: Structured sentiment analysis result
    """
    # End any existing run before starting a new one
    if mlflow.active_run():
        mlflow.end_run()

    # Create a descriptive run name
    run_name = f"sentiment_analysis_{company_name.replace(' ', '_')}"

    with mlflow.start_run(run_name=run_name):
        # Log the start time and company info
        mlflow.set_tag("analysis_type", "market_sentiment")
        mlflow.set_tag("input_company", company_name)

        result = market_sentiment_chain.invoke({"company_name": company_name})

        # Log additional metadata about the run
        mlflow.set_tag("status", "completed")

        return result
