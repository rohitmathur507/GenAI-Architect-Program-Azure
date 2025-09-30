import asyncio
from typing import List
import numpy as np
from tqdm.asyncio import tqdm

from config import config
from services.azure_client import azure_client
from models.data_models import BatchProcessingProgress, EmbeddingVector


class EmbeddingsService:
    """Handles generation of embeddings using Azure OpenAI"""

    def __init__(self):
        self.client = azure_client

    async def get_embedding(self, text: str) -> EmbeddingVector:
        """Generate embedding for a single text"""
        try:
            response = await self.client.embeddings.create(
                input=text, model=config.EMBEDDING_MODEL
            )
            return response.data[0].embedding
        except Exception as e:
            print(f"Error generating embedding: {e}")
            raise

    async def get_embeddings_batch(self, texts: List[str]) -> np.ndarray:
        """Generate embeddings for multiple texts in batches with progress tracking"""

        total_texts = len(texts)
        batch_size = config.EMBEDDING_BATCH_SIZE
        all_embeddings = []

        print("🎯 Generating embeddings from enhanced metadata...")

        # Create tasks for all texts
        async def get_embedding_task(text):
            return await self.get_embedding(text)

        # Process in batches with progress bar
        with tqdm(
            total=total_texts,
            desc="🔄 Embedding Generation",
            unit="texts",
            colour="yellow",
            dynamic_ncols=True,
        ) as pbar:

            for i in range(0, len(texts), batch_size):
                batch = texts[i : i + batch_size]

                # Create progress info
                current_batch = i // batch_size + 1
                total_batches = (len(texts) + batch_size - 1) // batch_size

                progress = BatchProcessingProgress(
                    total_items=total_texts,
                    current_batch=current_batch,
                    total_batches=total_batches,
                    batch_size=batch_size,
                    description="🔄 Embedding Batch",
                )

                pbar.set_description(progress.get_progress_description())

                # Execute batch
                tasks = [get_embedding_task(text) for text in batch]
                batch_embeddings = await asyncio.gather(*tasks)
                all_embeddings.extend(batch_embeddings)

                # Update progress
                pbar.update(len(batch))

        return np.array(all_embeddings, dtype="float32")

    async def get_query_embedding(self, query: str) -> np.ndarray:
        """Generate embedding for a search query"""
        embedding = await self.get_embedding(query)
        return np.array(embedding, dtype="float32")

    def normalize_embeddings(self, embeddings: np.ndarray) -> np.ndarray:
        """Normalize embeddings for cosine similarity"""
        import faiss

        normalized = embeddings.copy()
        faiss.normalize_L2(normalized)
        return normalized
