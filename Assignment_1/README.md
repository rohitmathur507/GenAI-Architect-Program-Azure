# Real-Time Market Sentiment Analyzer

A LangChain-powered pipeline that analyzes market sentiment for companies using recent financial news and Azure OpenAI.

## Features

- Company name to stock symbol extraction
- Real-time news fetching using Yahoo Finance
- AI-powered sentiment analysis with structured output
- MLflow integration for experiment tracking
- Streamlit web interface
- Modular, LCEL-based architecture

## Setup

### 1. Environment Variables

Create a `.env` file with your Azure OpenAI credentials:

```env
AZURE_OPENAI_ENDPOINT="your-endpoint"
AZURE_OPENAI_API_KEY="your-api-key"
OPENAI_API_VERSION="2024-02-15-preview"
AZURE_OPENAI_DEPLOYMENT="your-deployment-name"
OPENAI_API_TYPE="azure"
MLFLOW_TRACKING_URI="http://20.75.92.162:5000/"
```

### 2. Installation

```bash
pip install -r requirements.txt
```

Or using the project file:

```bash
pip install -e .
```

### 3. MLflow Setup

Ensure your MLflow tracking server is running at the specified URI. The application will automatically log all runs, parameters, and outputs.

## Usage

### Command Line Interface

```bash
python main.py
```

Enter a company name when prompted (e.g., "Apple Inc", "Microsoft Corporation").

### Streamlit Web Interface

```bash
streamlit run streamlit_app.py
```

Navigate to the provided URL and enter a company name in the web interface.

### Direct API Usage

```python
from chain_pipeline import analyze_market_sentiment

result = analyze_market_sentiment("Microsoft Corporation")
print(result)
```

## Output Format

```json
{
  "company_name": "Microsoft Corporation",
  "stock_code": "MSFT",
  "newsdesc": "Summary of recent news articles...",
  "sentiment": "Positive",
  "people_names": ["Satya Nadella", "Brad Smith"],
  "places_names": ["Redmond", "Washington"],
  "other_companies_referred": ["Google", "Amazon"],
  "related_industries": ["Technology", "Cloud Computing"],
  "market_implications": "Strong quarterly results indicate...",
  "confidence_score": 0.85
}
```

## Architecture

The application follows a modular LCEL chain architecture:

1. **Stock Symbol Extraction** (`stock_symbol.py`)
2. **News Fetching** (`news_fetcher.py`)  
3. **Sentiment Analysis** (`sentiment_analyzer.py`)
4. **MLflow Tracking** (`mlflow_tracker.py`)
5. **Chain Pipeline** (`chain_pipeline.py`)

## Sample Output - Microsoft Corporation

```json
{
  "company_name": "Microsoft Corporation",
  "stock_code": "MSFT", 
  "newsdesc": "Recent earnings report shows strong cloud growth with Azure revenue up 28%. CEO highlights AI integration across products. Stock reaches new 52-week high following quarterly results.",
  "sentiment": "Positive",
  "people_names": ["Satya Nadella"],
  "places_names": ["Redmond"],
  "other_companies_referred": ["Amazon", "Google"],
  "related_industries": ["Cloud Computing", "Artificial Intelligence", "Software"],
  "market_implications": "Strong fundamentals and AI positioning suggest continued growth trajectory with potential for market outperformance.",
  "confidence_score": 0.92
}
```

## MLflow Integration

All runs are automatically tracked with:
- Input parameters (company name, stock code)
- Output metrics (confidence score)
- Artifacts (news data, sentiment results)
- Tags (sentiment classification, model type)

Access the MLflow UI at your tracking URI to monitor experiments and compare results.
