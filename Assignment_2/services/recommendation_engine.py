from typing import List
import numpy as np
import pandas as pd

from config import config
from services.azure_client import azure_client
from services.embeddings_service import EmbeddingsService
from models.data_models import RecommendationResult, CompletedCourses, CourseMetadata


class RecommendationEngine:
    """Handles course recommendation logic"""

    def __init__(self):
        self.client = azure_client
        self.embeddings_service = EmbeddingsService()

    async def extract_completed_courses(
        self,
        profile: str,
        df: pd.DataFrame,
        index,
        enhanced_metadata: List[CourseMetadata],
    ) -> CompletedCourses:
        """Use vector database + LLM to extract completed courses efficiently"""

        if index is None:
            return []

        # Step 1: Use vector similarity to find potentially completed courses
        profile_embedding = await self.embeddings_service.get_query_embedding(profile)

        # Normalize and search for top similar courses
        profile_embedding = profile_embedding.reshape(1, -1).astype("float32")
        import faiss

        faiss.normalize_L2(profile_embedding)

        # Get top 5 most similar courses as candidates
        similarities, indices = index.search(
            profile_embedding, config.CANDIDATE_COURSES_LIMIT
        )

        candidate_courses = []
        for idx, similarity in zip(indices[0], similarities[0]):
            if idx < len(df):
                metadata = (
                    enhanced_metadata[idx] if enhanced_metadata else CourseMetadata()
                )
                candidate_courses.append(
                    {
                        "course_id": df.iloc[idx]["course_id"],
                        "title": df.iloc[idx]["title"],
                        "summary": metadata.summary
                        or df.iloc[idx]["description"][:100],
                        "domain": metadata.domain,
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
            response = await self.client.chat.completions.create(
                model=config.LLM_MODEL,
                messages=[
                    {
                        "role": "system",
                        "content": "You identify explicitly completed courses from user descriptions.",
                    },
                    {"role": "user", "content": prompt},
                ],
                temperature=config.LLM_TEMPERATURE,
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
        self,
        query_embedding: np.ndarray,
        completed_ids: CompletedCourses,
        df: pd.DataFrame,
        index,
        enhanced_metadata: List[CourseMetadata],
        k: int = 10,
    ) -> List[RecommendationResult]:
        """Retrieve top-k courses using FAISS with enhanced metadata"""

        if index is None:
            raise ValueError("Index not built. Build index first.")

        # Normalize query embedding
        query_embedding = query_embedding.reshape(1, -1).astype("float32")
        import faiss

        faiss.normalize_L2(query_embedding)

        # Search for top results (retrieve more than needed to account for filtering)
        search_k = min(k + len(completed_ids) + 10, len(df))
        similarities, indices = index.search(query_embedding, search_k)

        # Filter out completed courses and prepare results with enhanced metadata
        retrieved_courses = []
        for idx, similarity in zip(indices[0], similarities[0]):
            if idx >= len(df):  # Safety check
                continue

            course_id = df.iloc[idx]["course_id"]
            if course_id not in completed_ids:
                # Include enhanced metadata if available
                metadata = (
                    enhanced_metadata[idx] if enhanced_metadata else CourseMetadata()
                )

                result = RecommendationResult(
                    course_id=course_id,
                    title=df.iloc[idx]["title"],
                    description=df.iloc[idx]["description"],
                    summary=metadata.summary,
                    domain=metadata.domain,
                    key_skills=metadata.key_skills,
                    technologies=metadata.technologies,
                    difficulty=metadata.difficulty,
                    prerequisites=metadata.prerequisites,
                    similarity=float(similarity),
                )

                retrieved_courses.append(result)

                if len(retrieved_courses) == k:
                    break

        return retrieved_courses

    async def generate_recommendations(
        self, profile: str, retrieved_courses: List[RecommendationResult]
    ) -> str:
        """Use LLM to generate personalized recommendations with enhanced metadata"""

        # Prepare enhanced context from retrieved courses
        context = "\n\n".join(
            [
                f"Course ID: {course.course_id}\n"
                f"Title: {course.title}\n"
                f"Domain: {course.domain}\n"
                f"Difficulty: {course.difficulty}\n"
                f"Key Skills: {course.key_skills}\n"
                f"Technologies: {course.technologies}\n"
                f"Prerequisites: {course.prerequisites}\n"
                f"Summary: {course.summary or course.description}\n"
                f"Relevance Score: {course.similarity:.4f}"
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
            response = await self.client.chat.completions.create(
                model=config.LLM_MODEL,
                messages=[
                    {
                        "role": "system",
                        "content": "You are an expert course recommendation system that provides personalized learning paths using enhanced course metadata.",
                    },
                    {"role": "user", "content": prompt},
                ],
                temperature=config.RECOMMENDATION_TEMPERATURE,
                max_tokens=config.RECOMMENDATION_MAX_TOKENS,
            )
            return response.choices[0].message.content
        except Exception as e:
            raise Exception(f"Error generating LLM recommendations: {e}")
