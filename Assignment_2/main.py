import os
import asyncio
import pandas as pd
from dotenv import load_dotenv
from openai import AsyncAzureOpenAI
from typing import List, Tuple, Dict
import numpy as np
import faiss
from tqdm.asyncio import tqdm
import pickle
import hashlib
from pathlib import Path

load_dotenv()

client = AsyncAzureOpenAI(
    api_key=os.getenv("AZURE_OPENAI_API_KEY"),
    api_version=os.getenv("OPENAI_API_VERSION"),
    azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
)

EMBEDDING_MODEL = "text-embedding-3-small"
LLM_MODEL = "gpt4o"


class CourseRecommender:
    def __init__(self, catalog_path: str, vector_store_dir: str = "vector_store"):
        """Initialize the course recommender with dataset path and vector store directory"""
        try:
            self.df = pd.read_csv(catalog_path)
        except FileNotFoundError:
            raise FileNotFoundError(f"Dataset file not found: {catalog_path}")
        except Exception as e:
            raise Exception(f"Error loading dataset: {e}")

        self.catalog_path = catalog_path
        self.vector_store_dir = Path(vector_store_dir)
        self.vector_store_dir.mkdir(exist_ok=True)

        # Vector store file paths
        self.index_path = self.vector_store_dir / "faiss_index.bin"
        self.embeddings_path = self.vector_store_dir / "embeddings.pkl"
        self.metadata_path = self.vector_store_dir / "metadata.pkl"
        self.info_path = self.vector_store_dir / "store_info.pkl"

        self.embeddings = None
        self.index = None
        self.enhanced_metadata = None
        self.is_indexed = False

    def _get_dataset_hash(self) -> str:
        """Generate hash of dataset for versioning"""
        with open(self.catalog_path, "rb") as f:
            file_hash = hashlib.md5(f.read()).hexdigest()
        return file_hash

    def _save_vector_store(self):
        """Save vector store components to disk"""
        print("💾 Saving vector store to disk...")

        # Save FAISS index
        faiss.write_index(self.index, str(self.index_path))

        # Save embeddings
        with open(self.embeddings_path, "wb") as f:
            pickle.dump(self.embeddings, f)

        # Save metadata
        with open(self.metadata_path, "wb") as f:
            pickle.dump(self.enhanced_metadata, f)

        # Save store info for versioning
        store_info = {
            "dataset_hash": self._get_dataset_hash(),
            "total_courses": len(self.df),
            "embedding_model": EMBEDDING_MODEL,
            "llm_model": LLM_MODEL,
            "created_at": pd.Timestamp.now().isoformat(),
        }
        with open(self.info_path, "wb") as f:
            pickle.dump(store_info, f)

        print(f"✅ Vector store saved to {self.vector_store_dir}")

    def _load_vector_store(self) -> bool:
        """Load vector store components from disk. Returns True if successful."""
        try:
            # Check if all required files exist
            required_files = [
                self.index_path,
                self.embeddings_path,
                self.metadata_path,
                self.info_path,
            ]
            if not all(f.exists() for f in required_files):
                return False

            # Load and verify store info
            with open(self.info_path, "rb") as f:
                store_info = pickle.load(f)

            # Verify dataset hasn't changed
            current_hash = self._get_dataset_hash()
            if store_info["dataset_hash"] != current_hash:
                print(
                    "⚠️  Dataset has changed since vector store was created. Will regenerate..."
                )
                return False

            # Verify models haven't changed
            if (
                store_info.get("embedding_model") != EMBEDDING_MODEL
                or store_info.get("llm_model") != LLM_MODEL
            ):
                print("⚠️  Model configuration has changed. Will regenerate...")
                return False

            print("📂 Loading existing vector store...")

            # Load FAISS index
            self.index = faiss.read_index(str(self.index_path))

            # Load embeddings
            with open(self.embeddings_path, "rb") as f:
                self.embeddings = pickle.load(f)

            # Load metadata
            with open(self.metadata_path, "rb") as f:
                self.enhanced_metadata = pickle.load(f)

            self.is_indexed = True

            print(f"✅ Loaded vector store with {store_info['total_courses']} courses")
            print(f"📅 Created: {store_info['created_at']}")
            return True

        except Exception as e:
            print(f"⚠️  Failed to load vector store: {e}")
            return False

    def _clear_vector_store(self):
        """Clear existing vector store files"""
        files_to_remove = [
            self.index_path,
            self.embeddings_path,
            self.metadata_path,
            self.info_path,
        ]
        for file_path in files_to_remove:
            if file_path.exists():
                file_path.unlink()
        print("🗑️  Cleared existing vector store")

    async def extract_course_metadata(self, title: str, description: str) -> Dict:
        """Extract enhanced metadata from course title and description using LLM"""

        prompt = f"""Analyze this course and extract key metadata for better search and recommendations.

Course Title: {title}
Course Description: {description}

Extract and return the following metadata in a structured format:

1. Summary (2-3 sentences capturing the essence)
2. Prerequisites (what background knowledge is needed)
3. Key Skills (specific technical skills learned)
4. Technologies (tools, frameworks, languages covered)
5. Difficulty Level (Beginner/Intermediate/Advanced)
6. Domain (e.g., Machine Learning, Cloud Computing, Web Development)
7. Learning Outcomes (what you'll be able to do after completion)

Format your response as:
SUMMARY: [2-3 sentence summary]
PREREQUISITES: [list prerequisites]
KEY_SKILLS: [list key skills]
TECHNOLOGIES: [list technologies/tools]
DIFFICULTY: [level]
DOMAIN: [main domain]
OUTCOMES: [what learners will achieve]"""

        try:
            response = await client.chat.completions.create(
                model=LLM_MODEL,
                messages=[
                    {
                        "role": "system",
                        "content": "You are an expert course analyst who extracts meaningful metadata from course descriptions.",
                    },
                    {"role": "user", "content": prompt},
                ],
                temperature=0.1,
                max_tokens=500,
            )

            content = response.choices[0].message.content.strip()

            # Parse the structured response
            metadata = {}
            for line in content.split("\n"):
                if ":" in line:
                    key, value = line.split(":", 1)
                    key = key.strip().lower()
                    value = value.strip()
                    metadata[key] = value

            return metadata

        except Exception as e:
            print(f"Warning: Could not extract metadata for {title} - {e}")
            return {
                "summary": description,
                "prerequisites": "Not specified",
                "key_skills": "Various",
                "technologies": "Not specified",
                "difficulty": "Intermediate",
                "domain": "General",
                "outcomes": "Course completion",
            }

    async def compute_embeddings(self, refresh: bool = False):
        """
        Compute embeddings using enhanced metadata instead of raw descriptions

        Args:
            refresh (bool): If True, force regeneration of vector store even if it exists
        """
        # Try to load existing vector store first (unless refresh is requested)
        if not refresh and self._load_vector_store():
            return

        # Clear existing vector store if refresh is requested
        if refresh:
            self._clear_vector_store()

        print(
            "🚀 Starting enhanced course metadata extraction and embedding generation..."
        )

        total_courses = len(self.df)
        print(f"📚 Processing {total_courses} courses")

        # Extract enhanced metadata for all courses in parallel batches
        metadata_tasks = []
        for idx, row in self.df.iterrows():
            task = self.extract_course_metadata(row["title"], row["description"])
            metadata_tasks.append(task)

        # Process metadata extraction in batches with progress bar
        metadata_batch_size = 5  # Conservative for LLM API calls
        metadata_list = []

        print("🧠 Extracting course metadata using LLM...")

        # Create progress bar for metadata extraction
        with tqdm(
            total=total_courses,
            desc="📝 Metadata Extraction",
            unit="courses",
            colour="blue",
            dynamic_ncols=True,
        ) as pbar:

            for i in range(0, len(metadata_tasks), metadata_batch_size):
                batch_tasks = metadata_tasks[i : i + metadata_batch_size]

                # Update progress bar description with current batch
                current_batch = i // metadata_batch_size + 1
                total_batches = (
                    len(metadata_tasks) + metadata_batch_size - 1
                ) // metadata_batch_size
                pbar.set_description(
                    f"📝 Metadata Batch {current_batch}/{total_batches}"
                )

                batch_metadata = await asyncio.gather(*batch_tasks)
                metadata_list.extend(batch_metadata)

                # Update progress bar
                pbar.update(len(batch_tasks))

        # Create enhanced texts using extracted metadata
        enhanced_texts = []
        print("📋 Creating enhanced text representations...")

        with tqdm(
            total=total_courses,
            desc="🔧 Text Enhancement",
            unit="courses",
            colour="green",
            dynamic_ncols=True,
        ) as pbar:

            for idx, (metadata, row) in enumerate(
                zip(metadata_list, self.df.itertuples())
            ):
                enhanced_text = f"""
                Title: {row.title}
                Summary: {metadata.get('summary', '')}
                Domain: {metadata.get('domain', '')}
                Key Skills: {metadata.get('key_skills', '')}
                Technologies: {metadata.get('technologies', '')}
                Difficulty: {metadata.get('difficulty', '')}
                Prerequisites: {metadata.get('prerequisites', '')}
                Learning Outcomes: {metadata.get('outcomes', '')}
                """.strip()

                enhanced_texts.append(enhanced_text)
                pbar.update(1)

        self.enhanced_metadata = metadata_list

        print("🎯 Generating embeddings from enhanced metadata...")

        async def get_embedding(text):
            response = await client.embeddings.create(input=text, model=EMBEDDING_MODEL)
            return response.data[0].embedding

        # Process embeddings in batches with progress bar
        embedding_batch_size = 10  # Larger batch size for embedding API
        all_embeddings = []

        with tqdm(
            total=total_courses,
            desc="🔄 Embedding Generation",
            unit="courses",
            colour="yellow",
            dynamic_ncols=True,
        ) as pbar:

            for i in range(0, len(enhanced_texts), embedding_batch_size):
                batch = enhanced_texts[i : i + embedding_batch_size]

                # Update progress bar description with current batch
                current_batch = i // embedding_batch_size + 1
                total_batches = (
                    len(enhanced_texts) + embedding_batch_size - 1
                ) // embedding_batch_size
                pbar.set_description(
                    f"🔄 Embedding Batch {current_batch}/{total_batches}"
                )

                tasks = [get_embedding(text) for text in batch]
                batch_embeddings = await asyncio.gather(*tasks)
                all_embeddings.extend(batch_embeddings)

                # Update progress bar
                pbar.update(len(batch))

        self.embeddings = np.array(all_embeddings, dtype="float32")

        print("🏗️  Building FAISS index...")
        # Build FAISS index
        dimension = self.embeddings.shape[1]
        self.index = faiss.IndexFlatIP(dimension)

        # Normalize embeddings for cosine similarity
        faiss.normalize_L2(self.embeddings)
        self.index.add(self.embeddings)

        self.is_indexed = True

        # Save vector store to disk
        self._save_vector_store()

        print(f"✅ Successfully processed {total_courses} courses!")
        print(f"📊 Embedding dimension: {dimension}")
        print("🎯 FAISS index ready for recommendations")

    async def extract_completed_courses(self, profile: str) -> List[str]:
        """Use vector database + LLM to extract completed courses efficiently"""

        if not self.is_indexed:
            return []

        # Step 1: Use vector similarity to find potentially completed courses
        profile_embedding = await client.embeddings.create(
            input=profile, model=EMBEDDING_MODEL
        )
        query_embedding = np.array(profile_embedding.data[0].embedding)

        # Normalize and search for top similar courses
        query_embedding = query_embedding.reshape(1, -1).astype("float32")
        faiss.normalize_L2(query_embedding)

        # Get top 5 most similar courses as candidates
        similarities, indices = self.index.search(query_embedding, 5)

        candidate_courses = []
        for idx, similarity in zip(indices[0], similarities[0]):
            if idx < len(self.df):
                # Include enhanced metadata in candidates
                metadata = self.enhanced_metadata[idx] if self.enhanced_metadata else {}
                candidate_courses.append(
                    {
                        "course_id": self.df.iloc[idx]["course_id"],
                        "title": self.df.iloc[idx]["title"],
                        "summary": metadata.get(
                            "summary", self.df.iloc[idx]["description"][:100]
                        ),
                        "domain": metadata.get("domain", "General"),
                        "similarity": float(similarity),
                    }
                )

        # Step 2: Use LLM to determine which candidates are actually completed
        if not candidate_courses:
            return []

        candidates_text = "\n".join(
            [
                f"{course['course_id']}: {course['title']} - {course['domain']} ({course['summary'][:50]}...)"
                for course in candidate_courses
            ]
        )

        prompt = f"""Based on the user's description, identify which of these candidate courses they have explicitly completed, finished, or taken.

User Profile:
{profile}

Candidate Courses (most relevant to user's profile):
{candidates_text}

Look for explicit completion phrases like "I completed", "I finished", "I took", "I have done".
Return only the course IDs of courses the user explicitly states they completed.
If no explicit completions mentioned, return "NONE".

Response (course IDs only, one per line):"""

        try:
            response = await client.chat.completions.create(
                model=LLM_MODEL,
                messages=[
                    {
                        "role": "system",
                        "content": "You identify explicitly completed courses from user descriptions.",
                    },
                    {"role": "user", "content": prompt},
                ],
                temperature=0.1,
                max_tokens=100,
            )

            result = response.choices[0].message.content.strip()

            if result.upper() == "NONE":
                return []

            # Extract course IDs from response
            completed_ids = []
            for line in result.split("\n"):
                line = line.strip()
                if line and line.startswith("C") and len(line) <= 10:
                    completed_ids.append(line)

            return completed_ids

        except Exception as e:
            print(f"Warning: Could not extract completed courses - {e}")
            return []

    async def retrieve_top_k(
        self, query_embedding: np.ndarray, completed_ids: List[str], k: int = 10
    ) -> List[Dict]:
        """Retrieve top-k courses using FAISS with enhanced metadata"""
        if not self.is_indexed:
            raise ValueError("Index not built. Call compute_embeddings() first.")

        # Normalize query embedding
        query_embedding = query_embedding.reshape(1, -1).astype("float32")
        faiss.normalize_L2(query_embedding)

        # Search for top results (retrieve more than needed to account for filtering)
        search_k = min(k + len(completed_ids) + 10, len(self.df))
        similarities, indices = self.index.search(query_embedding, search_k)

        # Filter out completed courses and prepare results with enhanced metadata
        retrieved_courses = []
        for idx, similarity in zip(indices[0], similarities[0]):
            if idx >= len(self.df):  # Safety check
                continue

            course_id = self.df.iloc[idx]["course_id"]
            if course_id not in completed_ids:
                # Include enhanced metadata if available
                metadata = self.enhanced_metadata[idx] if self.enhanced_metadata else {}

                retrieved_courses.append(
                    {
                        "course_id": course_id,
                        "title": self.df.iloc[idx]["title"],
                        "description": self.df.iloc[idx]["description"],
                        "summary": metadata.get("summary", ""),
                        "domain": metadata.get("domain", "General"),
                        "key_skills": metadata.get("key_skills", ""),
                        "technologies": metadata.get("technologies", ""),
                        "difficulty": metadata.get("difficulty", "Intermediate"),
                        "prerequisites": metadata.get("prerequisites", ""),
                        "similarity": float(similarity),
                    }
                )
                if len(retrieved_courses) == k:
                    break

        return retrieved_courses

    async def generate_recommendations(
        self, profile: str, retrieved_courses: List[Dict]
    ) -> str:
        """Use LLM to generate personalized recommendations with enhanced metadata"""

        # Prepare enhanced context from retrieved courses
        context = "\n\n".join(
            [
                f"Course ID: {course['course_id']}\n"
                f"Title: {course['title']}\n"
                f"Domain: {course.get('domain', 'General')}\n"
                f"Difficulty: {course.get('difficulty', 'Intermediate')}\n"
                f"Key Skills: {course.get('key_skills', '')}\n"
                f"Technologies: {course.get('technologies', '')}\n"
                f"Prerequisites: {course.get('prerequisites', '')}\n"
                f"Summary: {course.get('summary', course.get('description', ''))}\n"
                f"Relevance Score: {course['similarity']:.4f}"
                for course in retrieved_courses
            ]
        )

        prompt = f"""You are a course recommendation expert. Based on the user's profile and the retrieved relevant courses with enhanced metadata, provide the top 5 course recommendations.

User Profile:
{profile}

Retrieved Relevant Courses (with enhanced metadata):
{context}

Task:
1. Analyze the user's background, interests, and learning goals
2. Consider the domain, difficulty level, prerequisites, and key skills for each course
3. Select and rank the top 5 most suitable courses based on learning progression
4. For each recommendation, explain why it's a good fit considering the metadata

Output Format (strictly follow):
RANK 1: [COURSE_ID]
Title: [Course Title]
Domain: [Domain]
Difficulty: [Difficulty Level]
Reasoning: [Brief explanation considering user background, course difficulty, and skill progression]

RANK 2: [COURSE_ID]
...

Provide exactly 5 recommendations."""

        try:
            response = await client.chat.completions.create(
                model=LLM_MODEL,
                messages=[
                    {
                        "role": "system",
                        "content": "You are an expert course recommendation system that provides personalized learning paths using enhanced course metadata.",
                    },
                    {"role": "user", "content": prompt},
                ],
                temperature=0.3,
                max_tokens=1500,
            )
            return response.choices[0].message.content
        except Exception as e:
            raise Exception(f"Error generating LLM recommendations: {e}")

    async def recommend_courses(self, profile: str) -> List[Tuple[str, float]]:
        """
        Autonomous recommendation function using vector-first course detection
        Returns: List of (course_id, similarity_score) for the top-5 recommendations
        """
        if not self.is_indexed:
            raise ValueError("Index not built. Call compute_embeddings() first.")

        # Step 1: Auto-extract completed courses from profile
        completed_ids = await self.extract_completed_courses(profile)

        # Step 2: Generate embedding for user profile
        response = await client.embeddings.create(input=profile, model=EMBEDDING_MODEL)
        query_embedding = np.array(response.data[0].embedding)

        # Step 3: Retrieve top-k relevant courses using FAISS
        retrieved_courses = await self.retrieve_top_k(
            query_embedding, completed_ids, k=10
        )

        # Step 4: Return top 5 as (course_id, similarity_score) tuples
        top_5_recommendations = [
            (course["course_id"], course["similarity"])
            for course in retrieved_courses[:5]
        ]

        return top_5_recommendations

    async def recommend_courses_detailed(self, profile: str) -> Tuple[List[Dict], str]:
        """
        Full autonomous RAG pipeline using vector-first detection and detailed analysis
        Returns: (retrieved_courses, llm_response)
        """
        if not self.is_indexed:
            raise ValueError("Index not built. Call compute_embeddings() first.")

        # Step 1: Auto-extract completed courses from profile
        completed_ids = await self.extract_completed_courses(profile)

        # Step 2: Generate embedding for user profile
        response = await client.embeddings.create(input=profile, model=EMBEDDING_MODEL)
        query_embedding = np.array(response.data[0].embedding)

        # Step 3: Retrieve top-k relevant courses using FAISS
        retrieved_courses = await self.retrieve_top_k(
            query_embedding, completed_ids, k=10
        )

        # Step 4: Generate personalized recommendations using LLM
        llm_recommendations = await self.generate_recommendations(
            profile, retrieved_courses
        )

        return retrieved_courses, llm_recommendations

    def get_vector_store_info(self) -> Dict:
        """Get information about the current vector store"""
        if not self.info_path.exists():
            return {"exists": False}

        try:
            with open(self.info_path, "rb") as f:
                store_info = pickle.load(f)

            store_info["exists"] = True
            store_info["is_current"] = (
                store_info["dataset_hash"] == self._get_dataset_hash()
                and store_info.get("embedding_model") == EMBEDDING_MODEL
                and store_info.get("llm_model") == LLM_MODEL
            )
            return store_info
        except Exception:
            return {"exists": False}

    def clear_vector_store(self):
        """Public method to clear vector store"""
        self._clear_vector_store()
        self.is_indexed = False
        self.embeddings = None
        self.index = None
        self.enhanced_metadata = None


async def run_evaluation(refresh_vector_store: bool = False):
    """
    Test the recommender with sample queries and generate evaluation report

    Args:
        refresh_vector_store (bool): If True, force regeneration of vector store
    """

    dataset_path = "assignment2dataset.csv"

    try:
        recommender = CourseRecommender(dataset_path)

        # Show vector store status
        store_info = recommender.get_vector_store_info()
        print("VECTOR STORE STATUS")
        print("=" * 50)
        if store_info["exists"]:
            print(f"📁 Vector store exists: {store_info['exists']}")
            print(f"📅 Created: {store_info.get('created_at', 'Unknown')}")
            print(f"📊 Courses: {store_info.get('total_courses', 'Unknown')}")
            print(f"🎯 Current: {store_info['is_current']}")
            if refresh_vector_store:
                print("🔄 Forcing refresh...")
        else:
            print("📁 No existing vector store found")
        print("=" * 50)

        await recommender.compute_embeddings(refresh=refresh_vector_store)
    except Exception as e:
        print(f"Failed to initialize recommender: {e}")
        return

    test_profiles = [
        "I've completed the 'Python Programming for Data Science' course and enjoy data visualization. What should I take next?",
        "I know Azure basics and want to manage containers and build CI/CD pipelines. Recommend courses.",
        "My background is in ML fundamentals; I'd like to specialize in neural networks and production workflows.",
        "I want to learn to build and deploy microservices with Kubernetes—what courses fit best?",
        "I'm interested in blockchain and smart contracts but have no prior experience. Which courses do you suggest?",
    ]

    print("\nCOURSE RECOMMENDATION ENGINE - EVALUATION REPORT")
    print("=" * 80)
    print(f"Dataset: {dataset_path}")
    print(f"Total courses: {len(recommender.df)}")
    print("=" * 80)

    for i, profile in enumerate(test_profiles, 1):
        print(f"\nTEST CASE {i}")
        print(f"USER QUERY: {profile}")
        print("-" * 80)

        try:
            basic_recommendations = await recommender.recommend_courses(profile)

            print("RECOMMENDATIONS:")
            for course_id, score in basic_recommendations:
                course_info = recommender.df[
                    recommender.df["course_id"] == course_id
                ].iloc[0]
                print(f"  ({course_id}, {score:.4f}) - {course_info['title']}")

        except Exception as e:
            print(f"ERROR: Failed to generate recommendations - {e}")

        print("=" * 80)


async def main():
    """Main function with command line argument support"""
    import sys

    # Simple command line argument parsing
    refresh = "--refresh" in sys.argv or "-r" in sys.argv

    if refresh:
        print("🔄 Refresh mode enabled - will regenerate vector store")

    await run_evaluation(refresh_vector_store=refresh)


if __name__ == "__main__":
    asyncio.run(main())
