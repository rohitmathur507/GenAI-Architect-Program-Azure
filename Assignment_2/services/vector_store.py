"""
Vector store operations using FAISS
"""

import pickle
import hashlib
from pathlib import Path
from typing import List, Tuple, Optional
import numpy as np
import faiss
import pandas as pd

from config import config
from models.data_models import VectorStoreInfo, CourseMetadata


class VectorStore:
    """Manages FAISS vector store operations"""

    def __init__(self, vector_store_dir: str = None):
        self.vector_store_dir = Path(vector_store_dir or config.VECTOR_STORE_DIR)
        self.vector_store_dir.mkdir(exist_ok=True)

        # Get file paths from config
        self.paths = config.get_vector_store_paths(str(self.vector_store_dir))

        self.index: Optional[faiss.Index] = None
        self.embeddings: Optional[np.ndarray] = None
        self.enhanced_metadata: Optional[List[CourseMetadata]] = None
        self.is_indexed = False

    def get_dataset_hash(self, dataset_path: str) -> str:
        """Generate hash of dataset for versioning"""
        with open(dataset_path, "rb") as f:
            file_hash = hashlib.md5(f.read()).hexdigest()
        return file_hash

    def save_vector_store(self, df: pd.DataFrame, dataset_path: str):
        """Save vector store components to disk"""
        print("💾 Saving vector store to disk...")

        # Save FAISS index
        faiss.write_index(self.index, str(self.paths["index"]))

        # Save embeddings
        with open(self.paths["embeddings"], "wb") as f:
            pickle.dump(self.embeddings, f)

        # Save metadata
        metadata_dicts = [meta.to_dict() for meta in self.enhanced_metadata]
        with open(self.paths["metadata"], "wb") as f:
            pickle.dump(metadata_dicts, f)

        # Save store info for versioning
        store_info = VectorStoreInfo(
            dataset_hash=self.get_dataset_hash(dataset_path),
            total_courses=len(df),
            embedding_model=config.EMBEDDING_MODEL,
            llm_model=config.LLM_MODEL,
            created_at=pd.Timestamp.now().isoformat(),
        )

        with open(self.paths["info"], "wb") as f:
            pickle.dump(
                (
                    store_info.to_dict()
                    if hasattr(store_info, "to_dict")
                    else store_info.__dict__
                ),
                f,
            )

        print(f"✅ Vector store saved to {self.vector_store_dir}")

    def load_vector_store(self, dataset_path: str) -> bool:
        """Load vector store components from disk. Returns True if successful."""
        try:
            # Check if all required files exist
            if not all(path.exists() for path in self.paths.values()):
                return False

            # Load and verify store info
            with open(self.paths["info"], "rb") as f:
                store_info_dict = pickle.load(f)

            store_info = VectorStoreInfo.from_dict(store_info_dict)

            # Verify dataset hasn't changed
            current_hash = self.get_dataset_hash(dataset_path)
            if store_info.dataset_hash != current_hash:
                print(
                    "⚠️  Dataset has changed since vector store was created. Will regenerate..."
                )
                return False

            # Verify models haven't changed
            if (
                store_info.embedding_model != config.EMBEDDING_MODEL
                or store_info.llm_model != config.LLM_MODEL
            ):
                print("⚠️  Model configuration has changed. Will regenerate...")
                return False

            print("📂 Loading existing vector store...")

            # Load FAISS index
            self.index = faiss.read_index(str(self.paths["index"]))

            # Load embeddings
            with open(self.paths["embeddings"], "rb") as f:
                self.embeddings = pickle.load(f)

            # Load metadata
            with open(self.paths["metadata"], "rb") as f:
                metadata_dicts = pickle.load(f)
                self.enhanced_metadata = [
                    CourseMetadata.from_dict(meta) for meta in metadata_dicts
                ]

            self.is_indexed = True

            print(f"✅ Loaded vector store with {store_info.total_courses} courses")
            print(f"📅 Created: {store_info.created_at}")
            return True

        except Exception as e:
            print(f"⚠️  Failed to load vector store: {e}")
            return False

    def clear_vector_store(self):
        """Clear existing vector store files"""
        for file_path in self.paths.values():
            if file_path.exists():
                file_path.unlink()
        print("🗑️  Cleared existing vector store")

        # Reset state
        self.index = None
        self.embeddings = None
        self.enhanced_metadata = None
        self.is_indexed = False

    def build_index(self, embeddings: np.ndarray):
        """Build FAISS index from embeddings"""
        print("🏗️  Building FAISS index...")

        # Store embeddings
        self.embeddings = embeddings.copy()

        # Build FAISS index
        dimension = embeddings.shape[1]
        self.index = faiss.IndexFlatIP(dimension)

        # Normalize embeddings for cosine similarity
        faiss.normalize_L2(self.embeddings)
        self.index.add(self.embeddings)

        self.is_indexed = True

        print(f"📊 Embedding dimension: {dimension}")
        print("🎯 FAISS index ready for recommendations")

    def search(
        self, query_embedding: np.ndarray, k: int = 10
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Search for similar embeddings"""
        if not self.is_indexed:
            raise ValueError("Index not built. Build index first.")

        # Normalize query embedding
        query_embedding = query_embedding.reshape(1, -1).astype("float32")
        faiss.normalize_L2(query_embedding)

        # Search
        similarities, indices = self.index.search(query_embedding, k)
        return similarities, indices

    def get_vector_store_info(self, dataset_path: str) -> VectorStoreInfo:
        """Get information about the current vector store"""
        if not self.paths["info"].exists():
            return VectorStoreInfo(
                dataset_hash="",
                total_courses=0,
                embedding_model="",
                llm_model="",
                created_at="",
                exists=False,
                is_current=False,
            )

        try:
            with open(self.paths["info"], "rb") as f:
                store_info_dict = pickle.load(f)

            store_info = VectorStoreInfo.from_dict(store_info_dict)
            store_info.exists = True
            store_info.is_current = (
                store_info.dataset_hash == self.get_dataset_hash(dataset_path)
                and store_info.embedding_model == config.EMBEDDING_MODEL
                and store_info.llm_model == config.LLM_MODEL
            )

            return store_info

        except Exception:
            return VectorStoreInfo(
                dataset_hash="",
                total_courses=0,
                embedding_model="",
                llm_model="",
                created_at="",
                exists=False,
                is_current=False,
            )
