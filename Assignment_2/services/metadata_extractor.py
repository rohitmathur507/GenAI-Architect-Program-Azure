import asyncio
from typing import Dict, List
from tqdm.asyncio import tqdm

from config import config
from services.azure_client import azure_client
from models.data_models import CourseMetadata, BatchProcessingProgress


class MetadataExtractor:
    """Handles extraction of enhanced metadata from course descriptions using LLM"""

    def __init__(self):
        self.client = azure_client

    async def extract_course_metadata(
        self, title: str, description: str
    ) -> CourseMetadata:
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
            response = await self.client.chat.completions.create(
                model=config.LLM_MODEL,
                messages=[
                    {
                        "role": "system",
                        "content": "You are an expert course analyst who extracts meaningful metadata from course descriptions.",
                    },
                    {"role": "user", "content": prompt},
                ],
                temperature=config.LLM_TEMPERATURE,
                max_tokens=config.LLM_MAX_TOKENS,
            )

            content = response.choices[0].message.content.strip()
            metadata_dict = self._parse_llm_response(content)
            return CourseMetadata.from_dict(metadata_dict)

        except Exception as e:
            print(f"Warning: Could not extract metadata for {title} - {e}")
            return CourseMetadata(
                summary=description,
                prerequisites="Not specified",
                key_skills="Various",
                technologies="Not specified",
                difficulty="Intermediate",
                domain="General",
                outcomes="Course completion",
            )

    def _parse_llm_response(self, content: str) -> Dict[str, str]:
        """Parse the structured LLM response into a dictionary"""
        metadata = {}
        for line in content.split("\n"):
            if ":" in line:
                key, value = line.split(":", 1)
                key = key.strip().lower()
                value = value.strip()
                metadata[key] = value
        return metadata

    async def extract_metadata_batch(
        self, courses_data: List[tuple]
    ) -> List[CourseMetadata]:
        """Extract metadata for multiple courses in batches with progress tracking"""

        total_courses = len(courses_data)
        batch_size = config.METADATA_BATCH_SIZE
        metadata_list = []

        print("🧠 Extracting course metadata using LLM...")

        # Create tasks for all courses
        tasks = [
            self.extract_course_metadata(title, description)
            for title, description in courses_data
        ]

        # Process in batches with progress bar
        with tqdm(
            total=total_courses,
            desc="📝 Metadata Extraction",
            unit="courses",
            colour="blue",
            dynamic_ncols=True,
        ) as pbar:

            for i in range(0, len(tasks), batch_size):
                batch_tasks = tasks[i : i + batch_size]

                # Create progress info
                current_batch = i // batch_size + 1
                total_batches = (len(tasks) + batch_size - 1) // batch_size

                progress = BatchProcessingProgress(
                    total_items=total_courses,
                    current_batch=current_batch,
                    total_batches=total_batches,
                    batch_size=batch_size,
                    description="📝 Metadata Batch",
                )

                pbar.set_description(progress.get_progress_description())

                # Execute batch
                batch_metadata = await asyncio.gather(*batch_tasks)
                metadata_list.extend(batch_metadata)

                # Update progress
                pbar.update(len(batch_tasks))

        return metadata_list

    def create_enhanced_text(self, title: str, metadata: CourseMetadata) -> str:
        """Create enhanced text representation from title and metadata"""
        return f"""
        Title: {title}
        Summary: {metadata.summary}
        Domain: {metadata.domain}
        Key Skills: {metadata.key_skills}
        Technologies: {metadata.technologies}
        Difficulty: {metadata.difficulty}
        Prerequisites: {metadata.prerequisites}
        Learning Outcomes: {metadata.outcomes}
        """.strip()
