"""
Main CourseRecommender class - orchestrates all services
"""

from typing import List, Tuple
import pandas as pd
from tqdm.asyncio import tqdm

from config import config
from services.vector_store import VectorStore
from services.metadata_extractor import MetadataExtractor
from services.embeddings_service import EmbeddingsService
from services.recommendation_engine import RecommendationEngine
from models.data_models import VectorStoreInfo, RecommendationResult, RecommendationList


class CourseRecommender:
    """Main course recommender that orchestrates all services"""

    def __init__(self, catalog_path: str, vector_store_dir: str = None):
        """Initialize the course recommender with dataset path and vector store directory"""
        try:
            self.df = pd.read_csv(catalog_path)
        except FileNotFoundError:
            raise FileNotFoundError(f"Dataset file not found: {catalog_path}")
        except Exception as e:
            raise Exception(f"Error loading dataset: {e}")

        self.catalog_path = catalog_path

        # Initialize services
        self.vector_store = VectorStore(vector_store_dir)
        self.metadata_extractor = MetadataExtractor()
        self.embeddings_service = EmbeddingsService()
        self.recommendation_engine = RecommendationEngine()

    async def compute_embeddings(self, refresh: bool = False):
        """
        Compute embeddings using enhanced metadata instead of raw descriptions

        Args:
            refresh (bool): If True, force regeneration of vector store even if it exists
        """
        # Try to load existing vector store first (unless refresh is requested)
        if not refresh and self.vector_store.load_vector_store(self.catalog_path):
            return

        # Clear existing vector store if refresh is requested
        if refresh:
            self.vector_store.clear_vector_store()

        print(
            "🚀 Starting enhanced course metadata extraction and embedding generation..."
        )

        total_courses = len(self.df)
        print(f"📚 Processing {total_courses} courses")

        # Step 1: Extract enhanced metadata for all courses
        courses_data = [
            (row["title"], row["description"]) for _, row in self.df.iterrows()
        ]
        metadata_list = await self.metadata_extractor.extract_metadata_batch(
            courses_data
        )

        # Step 2: Create enhanced texts using extracted metadata
        enhanced_texts = []
        print("📋 Creating enhanced text representations...")

        with tqdm(
            total=total_courses,
            desc="🔧 Text Enhancement",
            unit="courses",
            colour="green",
            dynamic_ncols=True,
        ) as pbar:
            for _, row in self.df.iterrows():
                idx = row.name
                metadata = metadata_list[idx]
                enhanced_text = self.metadata_extractor.create_enhanced_text(
                    row["title"], metadata
                )
                enhanced_texts.append(enhanced_text)
                pbar.update(1)

        # Step 3: Generate embeddings from enhanced metadata
        embeddings = await self.embeddings_service.get_embeddings_batch(enhanced_texts)

        # Step 4: Build FAISS index
        self.vector_store.build_index(embeddings)
        self.vector_store.enhanced_metadata = metadata_list

        # Step 5: Save vector store to disk
        self.vector_store.save_vector_store(self.df, self.catalog_path)

        print(f"✅ Successfully processed {total_courses} courses!")

    async def recommend_courses(self, profile: str) -> RecommendationList:
        """
        Autonomous recommendation function using vector-first course detection
        Returns: List of (course_id, similarity_score) for the top-5 recommendations
        """
        if not self.vector_store.is_indexed:
            raise ValueError("Index not built. Call compute_embeddings() first.")

        # Step 1: Auto-extract completed courses from profile
        completed_ids = await self.recommendation_engine.extract_completed_courses(
            profile,
            self.df,
            self.vector_store.index,
            self.vector_store.enhanced_metadata,
        )

        # Step 2: Generate embedding for user profile
        query_embedding = await self.embeddings_service.get_query_embedding(profile)

        # Step 3: Retrieve top-k relevant courses using FAISS
        retrieved_courses = await self.recommendation_engine.retrieve_top_k(
            query_embedding,
            completed_ids,
            self.df,
            self.vector_store.index,
            self.vector_store.enhanced_metadata,
            k=config.DEFAULT_TOP_K,
        )

        # Step 4: Return top 5 as (course_id, similarity_score) tuples
        top_5_recommendations = [
            (course.course_id, course.similarity)
            for course in retrieved_courses[: config.DEFAULT_RECOMMENDATIONS]
        ]

        return top_5_recommendations

    async def recommend_courses_detailed(
        self, profile: str
    ) -> Tuple[List[RecommendationResult], str]:
        """
        Full autonomous RAG pipeline using vector-first detection and detailed analysis
        Returns: (retrieved_courses, llm_response)
        """
        if not self.vector_store.is_indexed:
            raise ValueError("Index not built. Call compute_embeddings() first.")

        # Step 1: Auto-extract completed courses from profile
        completed_ids = await self.recommendation_engine.extract_completed_courses(
            profile,
            self.df,
            self.vector_store.index,
            self.vector_store.enhanced_metadata,
        )

        # Step 2: Generate embedding for user profile
        query_embedding = await self.embeddings_service.get_query_embedding(profile)

        # Step 3: Retrieve top-k relevant courses using FAISS
        retrieved_courses = await self.recommendation_engine.retrieve_top_k(
            query_embedding,
            completed_ids,
            self.df,
            self.vector_store.index,
            self.vector_store.enhanced_metadata,
            k=config.DEFAULT_TOP_K,
        )

        # Step 4: Generate personalized recommendations using LLM
        llm_recommendations = await self.recommendation_engine.generate_recommendations(
            profile, retrieved_courses
        )

        return retrieved_courses, llm_recommendations

    def get_vector_store_info(self) -> VectorStoreInfo:
        """Get information about the current vector store"""
        return self.vector_store.get_vector_store_info(self.catalog_path)

    def clear_vector_store(self):
        """Public method to clear vector store"""
        self.vector_store.clear_vector_store()

    @property
    def is_indexed(self) -> bool:
        """Check if the recommender is ready for recommendations"""
        return self.vector_store.is_indexed
