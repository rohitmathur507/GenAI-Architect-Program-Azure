import streamlit as st

from chain_pipeline import analyze_company_sentiment

st.title("Real-Time Market Sentiment Analyzer")
st.write("Enter a company name to analyze market sentiment based on recent news.")

company_name = st.text_input(
    "Company Name", placeholder="e.g., Apple Inc, Microsoft Corporation"
)

if st.button("Analyze Sentiment"):
    if company_name.strip():
        with st.spinner("Analyzing market sentiment..."):
            try:
                result = analyze_company_sentiment(company_name)

                st.success("Analysis Complete!")

                col1, col2 = st.columns(2)

                with col1:
                    st.subheader("Basic Information")
                    st.write(f"**Company:** {result['company_name']}")
                    st.write(f"**Stock Code:** {result['stock_code']}")
                    st.write(f"**Sentiment:** {result['sentiment']}")
                    st.write(f"**Confidence:** {result['confidence_score']:.2f}")

                with col2:
                    st.subheader("Market Implications")
                    st.write(result["market_implications"])

                st.subheader("News Summary")
                st.write(result["newsdesc"])

                col3, col4 = st.columns(2)

                with col3:
                    if result["people_names"]:
                        st.subheader("Key People")
                        for person in result["people_names"]:
                            st.write(f"• {person}")

                    if result["other_companies_referred"]:
                        st.subheader("Related Companies")
                        for company in result["other_companies_referred"]:
                            st.write(f"• {company}")

                with col4:
                    if result["places_names"]:
                        st.subheader("Locations")
                        for place in result["places_names"]:
                            st.write(f"• {place}")

                    if result["related_industries"]:
                        st.subheader("Industries")
                        for industry in result["related_industries"]:
                            st.write(f"• {industry}")

                st.subheader("Full JSON Output")
                st.json(result)

            except Exception as e:
                st.error(f"Error: {str(e)}")
    else:
        st.warning("Please enter a company name")

st.sidebar.title("About")
st.sidebar.info(
    "This application uses LangChain and Azure OpenAI to analyze market sentiment "
    "based on recent financial news. All analysis runs are tracked using MLflow."
)
