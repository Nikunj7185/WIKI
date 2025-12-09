import json
from tqdm import tqdm

# --- IMPORT YOUR PIPELINE ---
from wiked import wik_ans
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_groq import ChatGroq


# --------------------------------------------------
# Configuration
# --------------------------------------------------

MODEL_NAME = "llama-3.1-8b-instant"
EMBEDDING_NAME = "sentence-transformers/all-MiniLM-L6-v2"

EVAL_FILE = "testing/eval_questions.json"
TOP_K = 4


# --------------------------------------------------
# Utility: Recall@K
# --------------------------------------------------

def recall_at_k(context_docs, expected_answer: str) -> bool:
    """
    Checks if the expected answer appears in any retrieved chunk.
    """
    expected_answer = expected_answer.lower()
    for doc in context_docs:
        if expected_answer in doc.page_content.lower():
            return True
    return False


# --------------------------------------------------
# Main Evaluation Loop
# --------------------------------------------------

def run_retrieval_eval():
    # Load evaluation questions
    with open(EVAL_FILE, "r") as f:
        eval_data = json.load(f)

    # Initialize models
    embedding = HuggingFaceEmbeddings(model_name=EMBEDDING_NAME)
    llm = ChatGroq(model=MODEL_NAME, temperature=0.0)

    success = 0
    failures = []

    print("\n🔍 Starting Retrieval Evaluation...\n")

    for item in tqdm(eval_data):
        question = item["question"]
        expected_answer = item["expected_answer"]

        try:
            # We only care about retrieval stage
            _, _, _, retrieved_docs = wik_ans(
                user_query=question,
                model=llm,
                embedding=embedding,
                k_results=TOP_K,
                return_docs=True  
            )

            hit = recall_at_k(retrieved_docs, expected_answer)

            if hit:
                success += 1
            else:
                failures.append({
                    "question": question,
                    "expected_answer": expected_answer
                })

        except Exception as e:
            failures.append({
                "question": question,
                "error": str(e)
            })

    total = len(eval_data)
    recall_score = success / total

    print("\n✅ Retrieval Evaluation Complete")
    print(f"Recall@{TOP_K}: {recall_score:.2f} ({success}/{total})")

    if failures:
        print("\n❌ Failures:")
        for f in failures:
            print(f"- {f}")

    return recall_score


# --------------------------------------------------
# Entrypoint
# --------------------------------------------------

if __name__ == "__main__":
    run_retrieval_eval()
