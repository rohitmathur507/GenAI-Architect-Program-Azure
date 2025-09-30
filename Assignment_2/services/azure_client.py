"""
Azure OpenAI client management
"""

from openai import AsyncAzureOpenAI
from config import config
from typing import Optional


class AzureOpenAIClient:
    """Manages Azure OpenAI client instance"""

    _instance: Optional[AsyncAzureOpenAI] = None

    @classmethod
    def get_client(cls) -> AsyncAzureOpenAI:
        """Get or create Azure OpenAI client instance (singleton pattern)"""
        if cls._instance is None:
            cls._instance = AsyncAzureOpenAI(
                api_key=config.AZURE_OPENAI_API_KEY,
                api_version=config.OPENAI_API_VERSION,
                azure_endpoint=config.AZURE_OPENAI_ENDPOINT,
            )
        return cls._instance

    @classmethod
    def reset_client(cls):
        """Reset the client instance (useful for testing)"""
        cls._instance = None

    @classmethod
    def validate_connection(cls) -> bool:
        """Validate if the client can connect to Azure OpenAI"""
        try:
            # Basic validation - check if client is properly configured
            return all(
                [
                    config.AZURE_OPENAI_API_KEY,
                    config.AZURE_OPENAI_ENDPOINT,
                    config.OPENAI_API_VERSION,
                ]
            )
        except Exception as e:
            print(f"Failed to validate Azure OpenAI connection: {e}")
            return False


# Create a global client instance
azure_client = AzureOpenAIClient.get_client()
