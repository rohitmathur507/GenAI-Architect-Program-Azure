import json

from chain_pipeline import analyze_company_sentiment


def main():
    company_name = input("Enter company name: ").strip()

    if not company_name:
        print("Company name cannot be empty")
        return

    try:
        result = analyze_company_sentiment(company_name)
        print("\n" + "=" * 50)
        print("COMPANY SENTIMENT ANALYSIS RESULT")
        print("=" * 50)
        print(json.dumps(result, indent=2))

    except Exception as e:
        print(f"Error analyzing sentiment: {e}")


if __name__ == "__main__":
    main()
