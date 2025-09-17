from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import Runnable

from llm_client import get_azure_llm

STOCK_LOOKUP = {
    "apple inc": "AAPL",
    "microsoft corporation": "MSFT",
    "amazon.com inc": "AMZN",
    "google": "GOOGL",
    "alphabet inc": "GOOGL",
    "tesla inc": "TSLA",
    "meta platforms inc": "META",
    "facebook": "META",
    "nvidia corporation": "NVDA",
    "netflix inc": "NFLX",
    "adobe inc": "ADBE",
}


class StockSymbolExtractor(Runnable):
    def invoke(self, inputs, config=None):
        company_name = inputs if isinstance(inputs, str) else inputs.get("company_name")
        normalized = company_name.lower().strip()

        if normalized in STOCK_LOOKUP:
            return STOCK_LOOKUP[normalized]

        llm = get_azure_llm()
        prompt = PromptTemplate(
            input_variables=["company_name"],
            template="""Extract the stock ticker symbol for the company: {company_name}
            
            Return only the ticker symbol (e.g., AAPL, MSFT, TSLA).
            If unsure, make your best guess based on the company name.""",
        )

        chain = prompt | llm | StrOutputParser()
        return chain.invoke({"company_name": company_name})


stock_symbol_extractor = StockSymbolExtractor()
