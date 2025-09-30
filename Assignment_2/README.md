# Course Recommender

An AI-powered course recommendation system that uses Azure OpenAI to provide personalized course suggestions based on your profile and learning goals.

## 🚀 Quick Start

```bash
python evaluation.py                 # Run course recommendations
python evaluation.py --help          # See all options
```

## Setup

### 1. Environment Setup
Create a `.env` file with your Azure OpenAI credentials:
```bash
AZURE_OPENAI_API_KEY=your_api_key
AZURE_OPENAI_ENDPOINT=your_endpoint
OPENAI_API_VERSION=2024-02-15-preview
```

### 2. Install Dependencies
```bash
pip install pandas numpy faiss-cpu openai python-dotenv tqdm
```

## Usage

### Running Recommendations

```bash
# Standard evaluation with test profiles
python evaluation.py

# Force rebuild of AI models (if needed)
python evaluation.py --refresh

# Detailed analysis with explanations
python evaluation.py --detailed

# Validate your setup
python evaluation.py --validate

# Show help
python evaluation.py --help
```

### Programmatic Usage

```python
import asyncio
from course_recommender import CourseRecommender

async def main():
    # Initialize the recommender
    recommender = CourseRecommender("assignment2dataset.csv")
    
    # Build the AI models (one-time setup)
    await recommender.compute_embeddings()
    
    # Get recommendations for your profile
    profile = "I know Python and want to learn machine learning"
    recommendations = await recommender.recommend_courses(profile)
    
    print("Recommended courses:", recommendations)

if __name__ == "__main__":
    asyncio.run(main())
```

## Configuration

Edit `config.py` to customize:
- AI models used
- Processing batch sizes
- File paths

## Features

- **Smart Analysis**: AI analyzes course content to understand skills, difficulty, and prerequisites
- **Personalized Recommendations**: Matches courses to your background and goals
- **Fast Performance**: Caches AI models for quick subsequent runs
- **Detailed Explanations**: Provides reasoning for each recommendation

## Troubleshooting

- **Missing credentials**: Make sure your `.env` file has valid Azure OpenAI credentials
- **Slow first run**: Initial setup builds AI models, subsequent runs are faster
- **Use `--validate`**: Checks your setup before running recommendations