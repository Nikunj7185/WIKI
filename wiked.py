from dotenv import load_dotenv
load_dotenv()

from pydantic import BaseModel, Field
from typing import List
from RnC import RequestAndChunk as chunker

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_groq import ChatGroq
from langchain_community.embeddings import HuggingFaceEmbeddings


# --------------------------------------------------
# Pydantic Models
# --------------------------------------------------

class TopicList(BaseModel):
    """Wikipedia topics relevant to the user's question."""
    topics: List[str] = Field(
        description=(
            "A list of 2–3 high-quality, specific Wikipedia page titles "
            "that are essential to answer the user's question. "
            "Always include the main subject."
        )
    )


class RewrittenQueries(BaseModel):
    """Optimized search queries based on the user's question."""
    queries: List[str] = Field(
        description="A list of 2–3 alternative, semantically varied search queries."
    )


# --------------------------------------------------
# Core RAG Pipeline
# --------------------------------------------------

def wik_ans(
    user_query: str,
    model,
    embedding,
    k_results: int = 4,
    return_docs: bool = False  # enables evaluation
):
    """
    Executes an LLM-orchestrated RAG pipeline.
    Optionally returns retrieved documents for evaluation.
    """

    # ---------- 1a. Topic Generation ----------
    t_model = model.with_structured_output(TopicList)

    topic_prompt = ChatPromptTemplate.from_messages([
        SystemMessage(
            "You are an expert search assistant. Generate 3–5 exact Wikipedia article titles "
            "that best provide context for answering the user's question. "
            "Return ONLY valid JSON conforming to the TopicList schema."
        ),
        HumanMessage(user_query)
    ])

    topic_chain = topic_prompt | t_model
    topic_resp = topic_chain.invoke({})
    initial_topics = topic_resp.topics

    print(f"[Topics] Initial: {initial_topics}")

    # ---------- 1b. Reflection-Based Topic Filtering ----------
    reflection_prompt = ChatPromptTemplate.from_messages([
        SystemMessage(
            f"The user asked: '{user_query}'. "
            f"You generated the topics: {initial_topics}. "
            "Select the TWO most essential topics. "
            "Always include the main subject."
        ),
        HumanMessage("Return only the final topic list.")
    ])

    reflection_chain = reflection_prompt | t_model
    reflection_resp = reflection_chain.invoke({})
    final_topics = reflection_resp.topics

    print(f"[Topics] Final: {final_topics}")

    # ---------- 2. Retrieval / Indexing ----------
    lib = chunker(final_topics, embedding)

    # ---------- 3. Query Rewriting ----------
    q_model = model.with_structured_output(RewrittenQueries)

    rewrite_prompt = ChatPromptTemplate.from_messages([
        SystemMessage(
            "You are a query rewriting assistant. Generate 2–3 semantically different "
            "queries that may retrieve better context."
        ),
        HumanMessage(user_query)
    ])

    rewrite_chain = rewrite_prompt | q_model
    rewrite_resp = rewrite_chain.invoke({})

    search_queries = list(set([user_query] + rewrite_resp.queries))
    print(f"[Queries] Expanded: {search_queries}")

    # ---------- 4. Similarity Search ----------
    all_docs = []
    for q in search_queries:
        docs = lib.similarity_search(q, k=k_results)
        all_docs.extend(docs)

    # Deduplicate chunks
    unique_docs = {doc.page_content: doc for doc in all_docs}
    final_context_docs = list(unique_docs.values())[:k_results]

    print(f"[Retrieval] Retrieved {len(final_context_docs)} chunks")

    context_text = "\n---\n".join(doc.page_content for doc in final_context_docs)
    retrieved_sources = list(
        {doc.metadata.get("source", "N/A") for doc in final_context_docs}
    )

    # ---------- 5. Answer Generation ----------
    final_prompt = ChatPromptTemplate.from_messages([
        SystemMessage(
            "You are a factual assistant. Use ONLY the provided CONTEXT to answer the question. "
            "If the answer is not present in the context, say so explicitly.\n\n"
            f"CONTEXT:\n{context_text}"
        ),
        HumanMessage(user_query)
    ])

    answer_chain = final_prompt | model
    final_resp = answer_chain.invoke({})

    if return_docs:
        return (
            final_resp.content,
            retrieved_sources,
            "LLM-Orchestrated RAG",
            final_context_docs
        )

    return (
        final_resp.content,
        retrieved_sources,
        "LLM-Orchestrated RAG"
    )


# --------------------------------------------------
# Local Test
# --------------------------------------------------

if __name__ == "__main__":
    EMBEDDING_MODEL = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )
    LLM = ChatGroq(model="llama-3.1-8b-instant", temperature=0.0)

    answer, sources, method = wik_ans(
        "spouse of Kit Harington",
        LLM,
        EMBEDDING_MODEL
    )

    print("\n==============================")
    print(f"Method: {method}")
    print(f"Answer: {answer}")
    print(f"Sources: {sources}")
