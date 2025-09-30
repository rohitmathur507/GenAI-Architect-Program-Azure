"""
Data models and type definitions for Course Recommender
"""

from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass
import pandas as pd


@dataclass
class CourseMetadata:
    """Enhanced metadata for a course"""

    summary: str = ""
    prerequisites: str = "Not specified"
    key_skills: str = "Various"
    technologies: str = "Not specified"
    difficulty: str = "Intermediate"
    domain: str = "General"
    outcomes: str = "Course completion"

    @classmethod
    def from_dict(cls, data: Dict[str, str]) -> "CourseMetadata":
        """Create CourseMetadata from dictionary"""
        return cls(
            summary=data.get("summary", ""),
            prerequisites=data.get("prerequisites", "Not specified"),
            key_skills=data.get("key_skills", "Various"),
            technologies=data.get("technologies", "Not specified"),
            difficulty=data.get("difficulty", "Intermediate"),
            domain=data.get("domain", "General"),
            outcomes=data.get("outcomes", "Course completion"),
        )

    def to_dict(self) -> Dict[str, str]:
        """Convert CourseMetadata to dictionary"""
        return {
            "summary": self.summary,
            "prerequisites": self.prerequisites,
            "key_skills": self.key_skills,
            "technologies": self.technologies,
            "difficulty": self.difficulty,
            "domain": self.domain,
            "outcomes": self.outcomes,
        }


@dataclass
class Course:
    """Represents a course with basic information"""

    course_id: str
    title: str
    description: str
    metadata: Optional[CourseMetadata] = None

    @classmethod
    def from_dataframe_row(
        cls, row: pd.Series, metadata: Optional[CourseMetadata] = None
    ) -> "Course":
        """Create Course from pandas DataFrame row"""
        return cls(
            course_id=row["course_id"],
            title=row["title"],
            description=row["description"],
            metadata=metadata,
        )


@dataclass
class RecommendationResult:
    """Result of a course recommendation"""

    course_id: str
    title: str
    description: str
    summary: str = ""
    domain: str = "General"
    key_skills: str = ""
    technologies: str = ""
    difficulty: str = "Intermediate"
    prerequisites: str = ""
    similarity: float = 0.0

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "RecommendationResult":
        """Create RecommendationResult from dictionary"""
        return cls(
            course_id=data.get("course_id", ""),
            title=data.get("title", ""),
            description=data.get("description", ""),
            summary=data.get("summary", ""),
            domain=data.get("domain", "General"),
            key_skills=data.get("key_skills", ""),
            technologies=data.get("technologies", ""),
            difficulty=data.get("difficulty", "Intermediate"),
            prerequisites=data.get("prerequisites", ""),
            similarity=data.get("similarity", 0.0),
        )

    def to_dict(self) -> Dict[str, Any]:
        """Convert RecommendationResult to dictionary"""
        return {
            "course_id": self.course_id,
            "title": self.title,
            "description": self.description,
            "summary": self.summary,
            "domain": self.domain,
            "key_skills": self.key_skills,
            "technologies": self.technologies,
            "difficulty": self.difficulty,
            "prerequisites": self.prerequisites,
            "similarity": self.similarity,
        }


@dataclass
class VectorStoreInfo:
    """Information about the vector store"""

    dataset_hash: str
    total_courses: int
    embedding_model: str
    llm_model: str
    created_at: str
    exists: bool = True
    is_current: bool = True

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "VectorStoreInfo":
        """Create VectorStoreInfo from dictionary"""
        return cls(
            dataset_hash=data.get("dataset_hash", ""),
            total_courses=data.get("total_courses", 0),
            embedding_model=data.get("embedding_model", ""),
            llm_model=data.get("llm_model", ""),
            created_at=data.get("created_at", ""),
            exists=data.get("exists", True),
            is_current=data.get("is_current", True),
        )


@dataclass
class RecommendationRequest:
    """Request for course recommendations"""

    profile: str
    top_k: int = 10
    num_recommendations: int = 5
    exclude_completed: bool = True


@dataclass
class BatchProcessingProgress:
    """Progress tracking for batch processing"""

    total_items: int
    current_batch: int
    total_batches: int
    batch_size: int
    description: str = ""

    def get_progress_description(self) -> str:
        """Get formatted progress description"""
        return f"{self.description} {self.current_batch}/{self.total_batches}"


# Type aliases for better code readability
EmbeddingVector = List[float]
SimilarityScore = float
CourseID = str
UserProfile = str
CompletedCourses = List[CourseID]
RecommendationPair = Tuple[CourseID, SimilarityScore]
RecommendationList = List[RecommendationPair]
