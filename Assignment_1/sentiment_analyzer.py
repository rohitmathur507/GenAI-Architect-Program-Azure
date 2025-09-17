from langchain_core.output_parsers import PydanticOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import Runnable

from llm_client import get_azure_llm
from models import SentimentOutput


class SentimentAnalyzer(Runnable):
    def __init__(self):
        self.llm = get_azure_llm()
        self.parser = PydanticOutputParser(pydantic_object=SentimentOutput)
        self.prompt = PromptTemplate(
            input_variables=["company_name", "stock_code", "news_data"],
            template="""Analyze the following news data for {company_name} ({stock_code}) and provide a comprehensive structured sentiment analysis.

News Data:
{news_data}

Please extract and analyze the following information carefully:

1. **Overall Sentiment**: Classify as Positive, Negative, or Neutral based on the news impact on the company
2. **People Names**: Extract all individual person names mentioned (CEOs, executives, analysts, officials, politicians, etc.)
3. **Places**: Extract all geographic locations mentioned (cities, countries, states, regions, headquarters locations)
4. **Other Companies**: List all other company names, organizations, or brands mentioned
5. **Related Industries**: Identify industries or sectors mentioned or implied
6. **Market Implications**: Provide a detailed summary of how this news might affect the stock price and market perception
7. **Confidence Score**: Rate your confidence in the sentiment analysis (0.0-1.0)

Instructions for entity extraction:
- For people_names: Look for titles like CEO, President, Director, Analyst, etc. followed by names
- For places_names: Look for city names, country names, regions, or business locations
- For other_companies_referred: Include subsidiaries, partners, competitors, or any business entities mentioned
- Be thorough but only include entities that are explicitly mentioned in the text

{format_instructions}""",
            partial_variables={
                "format_instructions": self.parser.get_format_instructions()
            },
        )
        self.chain = self.prompt | self.llm | self.parser

    def invoke(self, inputs, config=None):
        return self.chain.invoke(inputs)


sentiment_analyzer = SentimentAnalyzer()
