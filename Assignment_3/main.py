import os
import json
import asyncio
import operator
from typing import TypedDict, Annotated, List, Literal
from datetime import datetime
from contextlib import asynccontextmanager

from dotenv import load_dotenv
from langchain_openai import AzureOpenAIEmbeddings, AzureChatOpenAI
from langchain_core.messages import HumanMessage, BaseMessage
from langchain_core.tools import tool
from langgraph.graph import StateGraph, END
from langgraph.prebuilt import create_react_agent
from pinecone import Pinecone, ServerlessSpec
import mlflow

load_dotenv()

# Azure credentials
AZURE_OPENAI_ENDPOINT = os.getenv("AZURE_OPENAI_ENDPOINT")
AZURE_OPENAI_API_KEY = os.getenv("AZURE_OPENAI_API_KEY")
AZURE_OPENAI_API_VERSION = os.getenv("OPENAI_API_VERSION")
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI")

LLM_DEPLOYMENT = "gpt4o"
EMBEDDING_DEPLOYMENT = "text-embedding-3-small"

# Configure MLflow tracking
if MLFLOW_TRACKING_URI:
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    print(f"MLflow tracking URI set to: {MLFLOW_TRACKING_URI}")
else:
    print("MLflow will use local tracking (mlruns directory)")

# Initialize async clients
embeddings = AzureOpenAIEmbeddings(
    azure_endpoint=AZURE_OPENAI_ENDPOINT,
    api_key=AZURE_OPENAI_API_KEY,
    api_version=AZURE_OPENAI_API_VERSION,
    azure_deployment=EMBEDDING_DEPLOYMENT,
)

llm = AzureChatOpenAI(
    azure_endpoint=AZURE_OPENAI_ENDPOINT,
    api_key=AZURE_OPENAI_API_KEY,
    api_version=AZURE_OPENAI_API_VERSION,
    azure_deployment=LLM_DEPLOYMENT,
    temperature=0,
)

pc = Pinecone(api_key=PINECONE_API_KEY)
INDEX_NAME = "assignment3-agentic-rag-kb"


@asynccontextmanager
async def mlflow_run_context(run_name: str, **params):
    """Async context manager for MLflow runs with automatic cleanup"""
    mlflow.start_run(run_name=run_name)
    try:
        # Log initial parameters
        for key, value in params.items():
            mlflow.log_param(key, value)
        yield
    finally:
        mlflow.end_run()


class AgentState(TypedDict):
    messages: Annotated[List[BaseMessage], operator.add]
    question: str
    retrieved_docs: Annotated[List[dict], operator.add]
    initial_answer: str
    critique_decision: Literal["COMPLETE", "REFINE", ""]
    final_answer: str
    retrieved_doc_ids: List[str]


async def load_and_index_kb(json_path: str):
    """Load KB and index to Pinecone asynchronously"""
    with open(json_path, "r") as f:
        kb_data = json.load(f)

    # Check if index exists
    index_exists = INDEX_NAME in pc.list_indexes().names()

    if not index_exists:
        print(f"Creating new Pinecone index: {INDEX_NAME}")
        pc.create_index(
            name=INDEX_NAME,
            dimension=1536,
            metric="cosine",
            spec=ServerlessSpec(cloud="aws", region="us-east-1"),
        )

    index = pc.Index(INDEX_NAME)

    # Check if index already has documents to avoid re-indexing
    if index_exists:
        stats = index.describe_index_stats()
        if stats["total_vector_count"] > 0:
            print(
                f"Index '{INDEX_NAME}' already contains {stats['total_vector_count']} documents. Skipping re-indexing."
            )
            print(
                "Embeddings are generated from the combination of 'question' + 'answer_snippet' fields"
            )
            return

    print(f"Generating embeddings for {len(kb_data)} documents...")
    print(
        "Embeddings are created from the combination of 'question' + 'answer_snippet' fields"
    )

    # Batch embed for efficiency - combining question and answer_snippet
    texts = [f"{doc['question']} {doc['answer_snippet']}" for doc in kb_data]
    embeddings_list = await asyncio.gather(
        *[asyncio.to_thread(embeddings.embed_query, text) for text in texts]
    )

    vectors = []
    for doc, embedding in zip(kb_data, embeddings_list):
        vectors.append(
            {
                "id": doc["doc_id"],
                "values": embedding,
                "metadata": {
                    "question": doc["question"],
                    "answer_snippet": doc["answer_snippet"],
                    "source": doc["source"],
                    "confidence_indicator": doc["confidence_indicator"],
                    "last_updated": doc["last_updated"],
                },
            }
        )

    index.upsert(vectors=vectors)
    print(f"Successfully indexed {len(vectors)} documents to Pinecone")


# ============= TOOLS =============


@tool
async def retrieve_top_5_documents(question: str) -> str:
    """Retrieve top 5 most relevant documents from knowledge base.

    Args:
        question: The user's question to search for

    Returns:
        Formatted string with retrieved documents and their IDs
    """
    index = pc.Index(INDEX_NAME)

    query_embedding = await asyncio.to_thread(embeddings.embed_query, question)

    results = await asyncio.to_thread(
        index.query, vector=query_embedding, top_k=5, include_metadata=True
    )

    retrieved = []
    for match in results["matches"]:
        doc_info = (
            f"[{match['id']}] (Score: {match['score']:.3f})\n"
            f"Source: {match['metadata']['source']}\n"
            f"Content: {match['metadata']['answer_snippet']}\n"
        )
        retrieved.append(doc_info)

    return "\n---\n".join(retrieved)


@tool
async def retrieve_one_additional_document(question: str, exclude_ids: str) -> str:
    """Retrieve 1 additional document not in the exclusion list.

    Args:
        question: The user's question
        exclude_ids: Comma-separated list of document IDs to exclude

    Returns:
        One new document with its ID and content
    """
    index = pc.Index(INDEX_NAME)

    excluded = [id.strip() for id in exclude_ids.split(",")]

    query_embedding = await asyncio.to_thread(embeddings.embed_query, question)

    results = await asyncio.to_thread(
        index.query, vector=query_embedding, top_k=10, include_metadata=True
    )

    # Find first document not in exclusion list
    for match in results["matches"]:
        if match["id"] not in excluded:
            return (
                f"[{match['id']}] (Score: {match['score']:.3f})\n"
                f"Source: {match['metadata']['source']}\n"
                f"Content: {match['metadata']['answer_snippet']}"
            )

    return "No additional documents found"


# ============= AGENT NODES =============


def create_retriever_agent():
    """Retriever Agent: Fetches top-5 documents from KB"""
    system_prompt = """You are a retriever agent specialized in fetching relevant documents.

Your task:
1. Use retrieve_top_5_documents tool with the user's question
2. Return the retrieved documents exactly as provided by the tool
3. Extract and list the document IDs (e.g., KB001, KB002)

Be precise and thorough."""

    agent = create_react_agent(
        llm, tools=[retrieve_top_5_documents], prompt=system_prompt
    )
    return agent


def create_answer_agent():
    """Answer Agent: Generates initial answer with citations"""
    system_prompt = """You are an answer generation agent specialized in creating comprehensive responses.

Your task:
1. Read the retrieved documents provided in the conversation
2. Generate a detailed, thorough answer to the user's question
3. Include [KBxxx] citations for every specific claim (e.g., [KB001], [KB002])
4. Only use information from the provided documents
5. Be comprehensive and actionable

Format: Provide a well-structured answer with proper citations."""

    agent = create_react_agent(llm, tools=[], prompt=system_prompt)
    return agent


def create_critique_agent():
    """Critique Agent: Evaluates answer completeness"""
    system_prompt = """You are a critique agent specialized in evaluating answer quality.

Your task:
1. Evaluate if the generated answer fully addresses the question
2. Check for gaps, missing details, or insufficient depth
3. Make a decision: COMPLETE or REFINE

Criteria:
- COMPLETE: Answer is comprehensive, detailed, and fully addresses all aspects
- REFINE: Answer has gaps, lacks depth, or misses important aspects

Output: Respond with ONLY one word: "COMPLETE" or "REFINE"
Do not provide explanations, just the decision word."""

    agent = create_react_agent(llm, tools=[], prompt=system_prompt)
    return agent


def create_refinement_agent():
    """Refinement Agent: Retrieves additional doc and improves answer"""
    system_prompt = """You are a refinement agent specialized in improving incomplete answers.

Your task:
1. Use retrieve_one_additional_document to get ONE more relevant document
2. Provide the excluded document IDs from the previous retrieval
3. Generate an improved, more comprehensive answer using ALL documents
4. Include [KBxxx] citations for all claims
5. Address gaps from the previous answer

Be thorough and ensure the refined answer is complete."""

    agent = create_react_agent(
        llm, tools=[retrieve_one_additional_document], prompt=system_prompt
    )
    return agent


# ============= GRAPH NODES =============


async def retriever_node(state: AgentState) -> AgentState:
    """Retriever node using agent"""
    agent = create_retriever_agent()

    messages = [HumanMessage(content=f"Retrieve documents for: {state['question']}")]

    result = await agent.ainvoke({"messages": messages})

    # Extract document IDs and content
    response_content = result["messages"][-1].content

    # Parse doc IDs
    doc_ids = []
    for line in response_content.split("\n"):
        if line.strip().startswith("[KB") and "]" in line:
            doc_id = line.split("]")[0].strip("[")
            doc_ids.append(doc_id)

    return {"messages": result["messages"], "retrieved_doc_ids": doc_ids}


async def answer_node(state: AgentState) -> AgentState:
    """Answer node using agent"""
    agent = create_answer_agent()

    # Get retrieved docs from previous messages
    retrieved_context = ""
    for msg in state["messages"]:
        if "[KB" in str(msg.content):
            retrieved_context = msg.content
            break

    prompt = f"""Question: {state['question']}

Retrieved Documents:
{retrieved_context}

Generate a comprehensive answer with [KBxxx] citations."""

    messages = state["messages"] + [HumanMessage(content=prompt)]

    result = await agent.ainvoke({"messages": messages})

    initial_answer = result["messages"][-1].content

    return {"messages": result["messages"], "initial_answer": initial_answer}


async def critique_node(state: AgentState) -> AgentState:
    """Critique node using agent"""
    agent = create_critique_agent()

    prompt = f"""Question: {state['question']}

Generated Answer:
{state['initial_answer']}

Evaluate: Is this answer COMPLETE or does it need REFINE?
Respond with only one word."""

    messages = state["messages"] + [HumanMessage(content=prompt)]

    result = await agent.ainvoke({"messages": messages})

    decision = result["messages"][-1].content.strip().upper()

    if decision not in ["COMPLETE", "REFINE"]:
        decision = "REFINE"

    return {"messages": result["messages"], "critique_decision": decision}


async def refinement_node(state: AgentState) -> AgentState:
    """Refinement node using agent"""
    agent = create_refinement_agent()

    exclude_ids = ",".join(state["retrieved_doc_ids"])

    prompt = f"""Question: {state['question']}

Previous Answer (incomplete):
{state['initial_answer']}

Excluded Document IDs: {exclude_ids}

Retrieve ONE additional document and generate an improved answer."""

    messages = state["messages"] + [HumanMessage(content=prompt)]

    result = await agent.ainvoke({"messages": messages})

    refined_answer = result["messages"][-1].content

    return {"messages": result["messages"], "final_answer": refined_answer}


async def complete_node(state: AgentState) -> AgentState:
    """Complete node: Use initial answer as final"""
    return {"final_answer": state["initial_answer"]}


# ============= ROUTING =============


def route_after_critique(state: AgentState) -> Literal["refinement", "complete"]:
    """Route based on critique decision"""
    if state["critique_decision"] == "REFINE":
        return "refinement"
    return "complete"


# ============= GRAPH CONSTRUCTION =============


def build_agentic_rag_graph():
    """Build 4-agent autonomous RAG graph"""
    workflow = StateGraph(AgentState)

    workflow.add_node("retriever", retriever_node)
    workflow.add_node("answer", answer_node)
    workflow.add_node("critique", critique_node)
    workflow.add_node("refinement", refinement_node)
    workflow.add_node("complete", complete_node)

    workflow.set_entry_point("retriever")
    workflow.add_edge("retriever", "answer")
    workflow.add_edge("answer", "critique")

    workflow.add_conditional_edges(
        "critique",
        route_after_critique,
        {"refinement": "refinement", "complete": "complete"},
    )

    workflow.add_edge("refinement", END)
    workflow.add_edge("complete", END)

    return workflow.compile()


async def run_query(graph, question: str, run_id: str = None):
    """Execute query through agentic RAG graph"""

    async def _execute_query():
        """Internal function to execute the query"""
        initial_state = {
            "messages": [],
            "question": question,
            "retrieved_docs": [],
            "initial_answer": "",
            "critique_decision": "",
            "final_answer": "",
            "retrieved_doc_ids": [],
        }

        result = await graph.ainvoke(initial_state)

        print(f"\n{'='*80}")
        print(f"Question: {question}")
        print(f"Critique Decision: {result['critique_decision']}")
        print(f"\nFinal Answer:\n{result['final_answer']}")
        print("=" * 80)

        return result

    # Only use MLflow context if run_id is provided (for non-parallel execution)
    if run_id:
        async with mlflow_run_context(
            run_name=f"query_{run_id}",
            question=question,
            timestamp=datetime.now().isoformat(),
        ):
            return await _execute_query()
    else:
        return await _execute_query()


async def run_parallel_queries(graph, queries: List[str]):
    """Execute multiple queries in parallel with proper resource management"""
    print(f"Starting parallel execution of {len(queries)} queries...")

    async with mlflow_run_context(
        run_name="parallel_queries_batch",
        total_queries=len(queries),
        queries=queries,
        execution_mode="parallel",
        timestamp=datetime.now().isoformat(),
    ):
        # Execute queries in parallel without individual MLflow runs
        tasks = [
            run_query(graph, query) for query in queries
        ]  # No run_id = no individual MLflow runs
        results = await asyncio.gather(*tasks)

        # Log aggregate results
        complete_count = sum(1 for r in results if r["critique_decision"] == "COMPLETE")
        refine_count = sum(1 for r in results if r["critique_decision"] == "REFINE")

        mlflow.log_param("complete_answers", complete_count)
        mlflow.log_param("refined_answers", refine_count)

        # Log all final answers as a single artifact
        all_answers = {}
        for i, (query, result) in enumerate(zip(queries, results)):
            all_answers[f"query_{i+1}"] = {
                "question": query,
                "critique_decision": result["critique_decision"],
                "final_answer": result["final_answer"],
            }

        mlflow.log_dict(all_answers, "all_parallel_results.json")

        print(
            f"\nParallel execution completed: {complete_count} complete, {refine_count} refined"
        )

    return results


async def test_single_query():
    """Test with a single query first"""
    await load_and_index_kb("self_critique_loop_dataset.json")

    graph = build_agentic_rag_graph()

    # Set experiment first
    mlflow.set_experiment("rohit_mathur_assignment3")

    print("\n" + "=" * 80)
    print("TESTING SINGLE QUERY")
    print("=" * 80)

    # Test single query with individual MLflow run
    test_query = "What are best practices for caching?"
    result = await run_query(graph, test_query, run_id="test_single")

    print("\nTest completed successfully!")
    print(f"Final answer preview: {result['final_answer'][:200]}...")


async def main():
    await load_and_index_kb("self_critique_loop_dataset.json")

    graph = build_agentic_rag_graph()

    queries = [
        "What are best practices for caching?",
        "How should I set up CI/CD pipelines?",
        "What are performance tuning tips?",
        "How do I version my APIs?",
        "What should I consider for error handling?",
        "What is 24*7?",
    ]

    mlflow.set_experiment("rohit_mathur_assignment3")

    print("\n" + "=" * 80)
    print("EXECUTING QUERIES IN PARALLEL WITH LANGGRAPH AGENTS")
    print("=" * 80)

    await run_parallel_queries(graph, queries)

    print("\n" + "=" * 80)
    print("PARALLEL EXECUTION COMPLETE")
    print("=" * 80)


if __name__ == "__main__":
    # For testing, run single query first
    # asyncio.run(test_single_query())

    # For full execution, run main
    asyncio.run(main())
