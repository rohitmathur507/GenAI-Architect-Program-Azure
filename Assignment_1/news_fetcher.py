from langchain_community.tools import YahooFinanceNewsTool
from langchain_core.runnables import Runnable


class NewsFetcher(Runnable):
    def __init__(self):
        self.yahoo_finance_tool = YahooFinanceNewsTool()

    def invoke(self, inputs, config=None):
        stock_symbol = inputs if isinstance(inputs, str) else inputs.get("stock_code")

        try:
            # Use YahooFinanceNewsTool to fetch news
            news_result = self.yahoo_finance_tool.invoke(stock_symbol)

            if not news_result or news_result.strip() == "":
                return f"No recent news found for {stock_symbol}"

            return news_result

        except Exception as e:
            return f"Error fetching news for {stock_symbol}: {str(e)}"


news_fetcher = NewsFetcher()

# Alternative: Create a runnable directly from the tool
news_fetcher_runnable = YahooFinanceNewsTool()
