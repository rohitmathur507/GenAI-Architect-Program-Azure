import os
from pathlib import Path
from typing import Dict, Any
from dotenv import load_dotenv

# Load environment variables
load_dotenv()


class Config:
    """Configuration class for the Course Recommender system"""

    # Azure OpenAI Configuration
    AZURE_OPENAI_API_KEY = os.getenv("AZURE_OPENAI_API_KEY")
    AZURE_OPENAI_ENDPOINT = os.getenv("AZURE_OPENAI_ENDPOINT")
    OPENAI_API_VERSION = os.getenv("OPENAI_API_VERSION", "2024-02-15-preview")

    # Model Configuration
    EMBEDDING_MODEL = "text-embedding-3-small"
    LLM_MODEL = "gpt4o"

    # Vector Store Configuration
    VECTOR_STORE_DIR = "vector_store"
    VECTOR_STORE_FILES = {
        "index": "faiss_index.bin",
        "embeddings": "embeddings.pkl",
        "metadata": "metadata.pkl",
        "info": "store_info.pkl",
    }

    # Processing Configuration
    METADATA_BATCH_SIZE = 5  # Conservative for LLM API calls
    EMBEDDING_BATCH_SIZE = 10  # Larger batch size for embedding API

    # Recommendation Configuration
    DEFAULT_TOP_K = 10
    DEFAULT_RECOMMENDATIONS = 5
    CANDIDATE_COURSES_LIMIT = 5

    # LLM Parameters
    LLM_TEMPERATURE = 0.1
    LLM_MAX_TOKENS = 1000
    RECOMMENDATION_TEMPERATURE = 0.3
    RECOMMENDATION_MAX_TOKENS = 1500

    # Dataset Configuration
    DEFAULT_DATASET_PATH = "assignment2dataset.csv"

    @classmethod
    def validate_config(cls) -> Dict[str, Any]:
        """Validate configuration and return status"""
        status = {"valid": True, "errors": [], "warnings": []}

        # Check required environment variables
        required_env_vars = ["AZURE_OPENAI_API_KEY", "AZURE_OPENAI_ENDPOINT"]

        for var in required_env_vars:
            if not getattr(cls, var):
                status["valid"] = False
                status["errors"].append(f"Missing required environment variable: {var}")

        # Check if dataset file exists
        if not Path(cls.DEFAULT_DATASET_PATH).exists():
            status["warnings"].append(
                f"Dataset file not found: {cls.DEFAULT_DATASET_PATH}"
            )

        return status

    @classmethod
    def get_vector_store_paths(cls, base_dir: str = None) -> Dict[str, Path]:
        """Get vector store file paths"""
        base_path = Path(base_dir or cls.VECTOR_STORE_DIR)

        return {
            name: base_path / filename
            for name, filename in cls.VECTOR_STORE_FILES.items()
        }

    @classmethod
    def print_config_summary(cls):
        """Print configuration summary"""
        print("🔧 CONFIGURATION SUMMARY")
        print("=" * 50)
        print(f"Embedding Model: {cls.EMBEDDING_MODEL}")
        print(f"LLM Model: {cls.LLM_MODEL}")
        print(f"Vector Store Dir: {cls.VECTOR_STORE_DIR}")
        print(f"Metadata Batch Size: {cls.METADATA_BATCH_SIZE}")
        print(f"Embedding Batch Size: {cls.EMBEDDING_BATCH_SIZE}")
        print(f"Default Dataset: {cls.DEFAULT_DATASET_PATH}")
        print("=" * 50)


# Create a global config instance
config = Config()
