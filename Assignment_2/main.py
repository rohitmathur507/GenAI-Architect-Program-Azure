import asyncio
import sys


def print_help():
    """Print help and usage information"""
    print("🎯 Course Recommender - AI-powered course recommendation system")
    print("=" * 60)
    print("\n📋 USAGE:")
    print("  This is a simple wrapper. For full functionality, use evaluation.py")
    print()
    print("🚀 QUICK START:")
    print("  python evaluation.py                 # Run standard evaluation")
    print("  python evaluation.py --refresh       # Force rebuild vector store")
    print("  python evaluation.py --detailed      # Detailed test with LLM")
    print("  python evaluation.py --validate      # Validate setup")
    print()
    print("💻 PROGRAMMATIC USAGE:")
    print("  from course_recommender import CourseRecommender")
    print("  recommender = CourseRecommender('dataset.csv')")
    print("  await recommender.compute_embeddings()")
    print("  recommendations = await recommender.recommend_courses(profile)")
    print()
    print("📁 OTHER SCRIPTS:")
    print("  python example_usage.py              # See usage examples")
    print("  python test_modular.py               # Test module imports")
    print("=" * 60)


async def main():
    """Main function - delegates to evaluation script"""

    # Check for help flag
    if "--help" in sys.argv or "-h" in sys.argv:
        print_help()
        return

    # Import and run evaluation
    try:
        from evaluation import CourseRecommenderEvaluator

        print("🎯 Course Recommender - Main Entry Point")
        print("Delegating to evaluation script for comprehensive functionality...")
        print()

        # Parse arguments and delegate
        refresh = "--refresh" in sys.argv or "-r" in sys.argv

        evaluator = CourseRecommenderEvaluator()

        # Run basic evaluation (initialization + simple test)
        await evaluator.run_evaluation(refresh_vector_store=refresh)

        print(
            "\n💡 TIP: Use 'python evaluation.py --help' or 'python main.py --help' for more options"
        )

    except ImportError as e:
        print(f"❌ Error importing evaluation module: {e}")
        print("💡 Make sure all required modules are available")
    except Exception as e:
        print(f"❌ Error running course recommender: {e}")


if __name__ == "__main__":
    asyncio.run(main())
