from typing import List

from pydantic import BaseModel, Field


class SentimentOutput(BaseModel):
    company_name: str = Field(description="Name of the company")
    stock_code: str = Field(description="Stock ticker symbol")
    newsdesc: str = Field(description="Summary of news articles")
    sentiment: str = Field(description="Overall sentiment: Positive/Negative/Neutral")
    people_names: List[str] = Field(description="Names of people mentioned")
    places_names: List[str] = Field(description="Places mentioned")
    other_companies_referred: List[str] = Field(description="Other companies mentioned")
    related_industries: List[str] = Field(description="Related industries")
    market_implications: str = Field(description="Market implications summary")
    confidence_score: float = Field(description="Confidence score between 0.0 and 1.0")
