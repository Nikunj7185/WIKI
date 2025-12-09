# WIKED  
**LLM-Orchestrated Retrieval-Augmented Generation (RAG) Microservice for Factual QA**

---

## Overview

**WIKED** is a deterministic, multi-stage **Retrieval-Augmented Generation (RAG)** system designed to answer factual questions using **Wikipedia as the knowledge source**.

Large Language Models (LLMs) are used **only for semantic transformations** (topic extraction, topic filtering, and query rewriting).  
All control flow is **explicit, deterministic, and developer-defined**.

---

## Key Features

- LLM-guided topic extraction using structured (Pydantic) outputs  
- Reflection-based topic filtering to reduce retrieval noise  
- Query rewriting for improved semantic recall  
- Dense vector retrieval using FAISS with HuggingFace embeddings  
- Source-grounded answer generation  
- Retrieval evaluation using **Recall@k**  
- FastAPI service with lifecycle-managed model loading  
- Dockerized for deployment  

---

## System Architecture

User Query  
↓  
Topic Extraction (LLM, structured)  
↓  
Topic Reflection & Filtering  
↓  
Wikipedia Loading + Chunking  
↓  
FAISS Dense Index  
↓  
Query Rewriting (LLM)  
↓  
Multi-Query Similarity Search  
↓  
Deduplicated Context  
↓  
Answer Generation (LLM)

---

## Project Structure

```
WIKED/
├── app.py                 # FastAPI service
├── wiked.py               # Core RAG pipeline
├── RnC.py                 # Wikipedia loading + chunking + FAISS indexing
├── testing/
│   ├── retrieval_eval.py  # Recall@k evaluation
│   ├── eval_questions.json
│   └── __init__.py
├── requirements.txt
├── Dockerfile
├── __init__.py
└── README.md
```

---

## Core Design Decisions

### 1. Deterministic Orchestration (Non-Agentic)

- No autonomous control loops  
- No dynamic tool selection  
- No planner or policy execution  

LLMs are used **only** for:
- Topic extraction  
- Topic filtering  
- Query rewriting  

This keeps the system predictable, debuggable, and easy to evaluate.

---

### 2. Structured Outputs for Reliability

Intermediate LLM outputs are constrained using **Pydantic schemas**, ensuring:
- Valid JSON outputs  
- Stable downstream parsing  
- Reduced runtime failures  

---

### 3. Retrieval-First Philosophy

Retrieval quality is treated as a **first-class concern**.  
The system explicitly evaluates whether relevant information was retrieved *before* evaluating answer quality.

---

## Evaluation Methodology

### Metrics Used

#### Recall@k (Primary Metric)

Recall@k measures whether the correct answer appears in **any of the top-k retrieved chunks**.

Recall@k =  (number of queries where answer appears in retrieved context) / (total queries)

This metric isolates **retrieval failures** from **generation failures**.

---

### Evaluation Dataset

- 20 factual questions  
- Single-hop Wikipedia facts  
- Mix of entities, dates, definitions, and relationships  

Stored in:

```
testing/eval_questions.json
```

---

### Retrieval Evaluation Results

```
✅ Retrieval Evaluation Complete
Recall@4: 0.80 (16 / 20)
```

---

## Running the Evaluation

```
python -m testing.retrieval_eval
```

---

## API Usage

### Run Locally

```
uvicorn app:app --host 0.0.0.0 --port 8000
```

### Endpoint

```
POST /wiked_answer
```

---

## Docker Deployment

```
docker build -t wiked .
docker run -p 8000:8000 wiked
```

---

## Limitations

- FAISS index is rebuilt per query  
- Wikipedia pages are fetched per request  
- No reranking stage  
- No multi-hop reasoning  
- No caching layer  

---

## Planned Improvements

- Persistent vector store  
- Cross-encoder reranking  
- Faithfulness evaluation  
- Query caching  

---

## Author

**Nikunj Upadhyay**  
IIT (BHU) Varanasi  
Focus: Applied ML, LLM Systems, Retrieval Engineering
