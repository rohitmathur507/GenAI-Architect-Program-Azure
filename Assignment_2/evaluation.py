"""
Evaluation and testing script for Course Recommender
"""

import asyncio
import sys

from config import config
from course_recommender import CourseRecommender


class CourseRecommenderEvaluator:
    """Handles evaluation and testing of the course recommender"""

    def __init__(self, dataset_path: str = None):
        self.dataset_path = dataset_path or config.DEFAULT_DATASET_PATH
        self.test_profiles = [
            "I've completed the 'Python Programming for Data Science' course and enjoy data visualization. What should I take next?",
            "I know Azure basics and want to manage containers and build CI/CD pipelines. Recommend courses.",
            "My background is in ML fundamentals; I'd like to specialize in neural networks and production workflows.",
            "I want to learn to build and deploy microservices with Kubernetes—what courses fit best?",
            "I'm interested in blockchain and smart contracts but have no prior experience. Which courses do you suggest?",
        ]

    async def run_evaluation(self, refresh_vector_store: bool = False):
        """
        Test the recommender with sample queries and generate evaluation report

        Args:
            refresh_vector_store (bool): If True, force regeneration of vector store
        """
        try:
            recommender = CourseRecommender(self.dataset_path)

            # Show configuration
            config.print_config_summary()

            # Show vector store status
            self._print_vector_store_status(recommender, refresh_vector_store)

            # Initialize recommender
            await recommender.compute_embeddings(refresh=refresh_vector_store)

        except Exception as e:
            print(f"Failed to initialize recommender: {e}")
            return

        # Run evaluation tests
        await self._run_recommendation_tests(recommender)

    def _print_vector_store_status(
        self, recommender: CourseRecommender, refresh_vector_store: bool
    ):
        """Print vector store status information"""
        store_info = recommender.get_vector_store_info()

        print("\nVECTOR STORE STATUS")
        print("=" * 50)
        if store_info.exists:
            print(f"📁 Vector store exists: {store_info.exists}")
            print(f"📅 Created: {store_info.created_at or 'Unknown'}")
            print(f"📊 Courses: {store_info.total_courses or 'Unknown'}")
            print(f"🎯 Current: {store_info.is_current}")
            if refresh_vector_store:
                print("🔄 Forcing refresh...")
        else:
            print("📁 No existing vector store found")
        print("=" * 50)

    async def _run_recommendation_tests(self, recommender: CourseRecommender):
        """Run recommendation tests with all test profiles"""
        print("\nCOURSE RECOMMENDATION ENGINE - EVALUATION REPORT")
        print("=" * 80)
        print(f"Dataset: {self.dataset_path}")
        print(f"Total courses: {len(recommender.df)}")
        print("=" * 80)

        for i, profile in enumerate(self.test_profiles, 1):
            await self._run_single_test(recommender, i, profile)

    async def _run_single_test(
        self, recommender: CourseRecommender, test_num: int, profile: str
    ):
        """Run a single recommendation test"""
        print(f"\nTEST CASE {test_num}")
        print(f"USER QUERY: {profile}")
        print("-" * 80)

        try:
            # Get basic recommendations
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

    async def run_detailed_test(self, profile: str):
        """Run a detailed test with a single profile"""
        try:
            recommender = CourseRecommender(self.dataset_path)

            # Ensure vector store is ready
            if not recommender.is_indexed:
                print("Building vector store...")
                await recommender.compute_embeddings()

            print("\nDETAILED RECOMMENDATION TEST")
            print("=" * 80)
            print(f"USER QUERY: {profile}")
            print("-" * 80)

            # Get detailed recommendations
            retrieved_courses, llm_response = (
                await recommender.recommend_courses_detailed(profile)
            )

            print("\nRETRIEVED COURSES:")
            for i, course in enumerate(retrieved_courses[:5], 1):
                print(f"\n{i}. {course.title} ({course.course_id})")
                print(f"   Domain: {course.domain}")
                print(f"   Difficulty: {course.difficulty}")
                print(f"   Similarity: {course.similarity:.4f}")
                print(f"   Summary: {course.summary[:100]}...")

            print("\nLLM RECOMMENDATIONS:")
            print(llm_response)
            print("=" * 80)

        except Exception as e:
            print(f"ERROR: Failed to run detailed test - {e}")

    def validate_setup(self) -> bool:
        """Validate that the setup is correct"""
        print("🔍 SETUP VALIDATION")
        print("=" * 50)

        # Validate configuration
        config_status = config.validate_config()

        if not config_status["valid"]:
            print("❌ Configuration validation failed:")
            for error in config_status["errors"]:
                print(f"   - {error}")
            return False

        if config_status["warnings"]:
            print("⚠️  Configuration warnings:")
            for warning in config_status["warnings"]:
                print(f"   - {warning}")

        print("✅ Configuration is valid")
        print("=" * 50)
        return True


async def main():
    """Main function with command line argument support"""
    # Simple command line argument parsing
    refresh = "--refresh" in sys.argv or "-r" in sys.argv
    detailed = "--detailed" in sys.argv or "-d" in sys.argv
    validate = "--validate" in sys.argv or "-v" in sys.argv
    help_flag = "--help" in sys.argv or "-h" in sys.argv

    if help_flag:
        print_help()
        return

    evaluator = CourseRecommenderEvaluator()

    # Validate setup if requested
    if validate:
        if not evaluator.validate_setup():
            return

    if refresh:
        print("🔄 Refresh mode enabled - will regenerate vector store")

    if detailed:
        # Run detailed test with first profile
        test_profile = evaluator.test_profiles[0]
        await evaluator.run_detailed_test(test_profile)
    else:
        # Run standard evaluation
        await evaluator.run_evaluation(refresh_vector_store=refresh)


def print_help():
    """Print help information for evaluation script"""
    print("🧪 Course Recommender Evaluation & Testing Script")
    print("=" * 60)
    print("\n📋 USAGE:")
    print("  python evaluation.py [options]")
    print()
    print("🔧 OPTIONS:")
    print("  -h, --help      Show this help message")
    print("  -r, --refresh   Force regeneration of vector store")
    print("  -d, --detailed  Run detailed test with LLM recommendations")
    print("  -v, --validate  Validate setup and configuration")
    print()
    print("📊 EVALUATION MODES:")
    print("  Standard:  Tests all predefined profiles with basic recommendations")
    print("  Detailed:  Tests one profile with full LLM analysis and explanations")
    print("  Validate:  Checks configuration and setup without running tests")
    print()
    print("💡 EXAMPLES:")
    print("  python evaluation.py                # Standard evaluation")
    print("  python evaluation.py --refresh      # Rebuild vector store first")
    print("  python evaluation.py --detailed     # Detailed analysis")
    print("  python evaluation.py --validate     # Check setup only")
    print("=" * 60)


if __name__ == "__main__":
    asyncio.run(main())
