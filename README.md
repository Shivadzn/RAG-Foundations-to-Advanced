# RAG: Foundations to Advanced

A curated collection of 15 Jupyter notebooks covering Retrieval-Augmented Generation from first principles through agentic, self-corrective systems. Every concept is grounded in runnable code, LangChain/LangGraph implementations, and peer-reviewed research.

**Stack:** Python · LangChain · LangGraph · ChromaDB · Groq · LangSmith · RAGAS

---

<!-- IMAGES: Replace the commented lines below with your actual image tags once exported.
     Recommended workflow:
       1. Export each diagram as a PNG at 1400px wide (browser screenshot or draw.io export).
       2. Create a docs/assets/ folder in your repository root.
       3. Place images there and uncomment the lines.

     ![Learning Roadmap](docs/assets/roadmap.png)
     ![Pipeline Architecture](docs/assets/pipeline.png)
-->

## Table of Contents

- [Repository Structure](#repository-structure)
- [Notebook Index](#notebook-index)
- [Architecture Overview](#architecture-overview)
- [Notebook Deep-Dives](#notebook-deep-dives)
  - [Part 1 — Foundations](#part-1--foundations)
  - [Part 2 — Intermediate: Query Intelligence](#part-2--intermediate-query-intelligence)
  - [Part 3 — Advanced: Agentic and Self-Corrective RAG](#part-3--advanced-agentic-and-self-corrective-rag)
- [Setup](#setup)
- [Suggested Learning Paths](#suggested-learning-paths)
- [Research Papers Referenced](#research-papers-referenced)
- [Key Concepts Glossary](#key-concepts-glossary)
- [Contributing](#contributing)

---

## Repository Structure

```
RAG basic to Advanced/
├── Basic text splitting and chunking.ipynb      # 01 — Foundations
├── 1-intro-to-RAG.ipynb                         # 02
├── basic-to-advanced-BM25.ipynb                 # 03
├── Hybrid Approach_ BM25 + Vector Embeddings.ipynb  # 04
├── 2-Query-Transformations.ipynb                # 05 — Intermediate
├── 3-Routing-to-Datasources.ipynb               # 06
├── 4-Query-Construction.ipynb                   # 07
├── 5-Indexing-to-VectoreDB_s.ipynb              # 08
├── 6-Retrieval-Mechanisms.ipynb                 # 09
├── 7-Agentic-RAG.ipynb                          # 10 — Advanced
├── 8-Adaptive-RAG-Agent.ipynb                   # 11
├── 9-Corrective-RAG.ipynb                       # 12
├── 10-Local-RAG-Agent-with-Llama3.ipynb         # 13
├── Self-RAG.ipynb                               # 14
└── RAGAS_CODE.ipynb                             # 15 — Evaluation
```

---

## Notebook Index

| # | Notebook | Level | Core Topics |
|---|----------|-------|-------------|
| 1 | `Basic text splitting and chunking.ipynb` | Beginner | Character, recursive, token, and semantic chunking |
| 2 | `1-intro-to-RAG.ipynb` | Beginner | Document loading, chunking, vector store, async retrieval |
| 3 | `basic-to-advanced-BM25.ipynb` | Beginner | TF-IDF, BM25 scoring, IDF normalization, document-length penalty |
| 4 | `Hybrid Approach_ BM25 + Vector Embeddings.ipynb` | Beginner | Two-stage reranking, Cohere reranker, RRF |
| 5 | `2-Query-Transformations.ipynb` | Intermediate | Multi-query, RAG-fusion, step-back prompting, HyDE |
| 6 | `3-Routing-to-Datasources.ipynb` | Intermediate | Logical routing, semantic routing, function calling |
| 7 | `4-Query-Construction.ipynb` | Intermediate | Metadata filtering, structured query generation |
| 8 | `5-Indexing-to-VectoreDB_s.ipynb` | Intermediate | Multi-representation indexing, parent-document retrieval |
| 9 | `6-Retrieval-Mechanisms.ipynb` | Intermediate | RAG Fusion, RRF, re-ranking, CRAG retrieval |
| 10 | `7-Agentic-RAG.ipynb` | Advanced | LangGraph agent loop, state machines, conditional edges |
| 11 | `8-Adaptive-RAG-Agent.ipynb` | Advanced | Query analysis, no/single/iterative retrieval routing |
| 12 | `9-Corrective-RAG.ipynb` | Advanced | CRAG paper, relevance evaluation, web search fallback |
| 13 | `10-Local-RAG-Agent-with-Llama3.ipynb` | Advanced | Ollama, GPT4All embeddings, fully local pipeline |
| 14 | `Self-RAG.ipynb` | Advanced | Self-reflection, retrieval grading, hallucination detection |
| 15 | `RAGAS_CODE.ipynb` | Advanced | Faithfulness, answer relevancy, context recall metrics |

---

## Architecture Overview

The RAG pipeline runs in two phases.

**Indexing** (offline, run once)

```
Documents → Text Splitting → Embedding → Smart Indexing → Vector Store
                                     └─► BM25 Index (sparse)
```

**Query → Retrieval → Generation** (online, per request)

```
User Query → Query Transform → Retrieval → Self-Correction → LLM → Answer
                                               |
                              [CRAG / Self-RAG / Adaptive RAG]
```

**Evaluation** (continuous)

```
Answer + Context → RAGAS → Faithfulness · Answer Relevancy · Context Precision · Context Recall
```

---

## Notebook Deep-Dives

### Part 1 — Foundations

---

#### `Basic text splitting and chunking.ipynb`

Before anything else, documents must be broken into chunks small enough to embed meaningfully. This notebook covers four strategies in order of increasing sophistication:

| Strategy | Mechanism | Best For |
|----------|-----------|----------|
| Character splitting | Fixed character count, no semantic awareness | Quick experiments, uniform text |
| Recursive character splitting | Tries `\n\n`, `\n`, ` ` in sequence; preserves paragraphs | General-purpose documents |
| Token-based splitting | Splits on token boundaries using tiktoken | Precise LLM context-window alignment |
| Semantic splitting | Measures embedding similarity between sentences; splits at low-similarity boundaries | High-quality retrieval, varied content |

`chunk_overlap` (typically 50–200 characters) prevents information loss at chunk boundaries — the trailing context of one chunk seeds the next.

---

#### `1-intro-to-RAG.ipynb`

The canonical RAG loop from scratch.

```
1. Load web document    →  WebBaseLoader
2. Split into chunks    →  RecursiveCharacterTextSplitter
3. Embed and store      →  OpenAI Embeddings → ChromaDB
4. Retrieve context     →  Similarity search
5. Generate answer      →  Groq LLM via async ainvoke
```

Uses `asyncio.gather()` so multiple queries fire in parallel, bypassing the serial bottleneck of synchronous API calls.

---

#### `basic-to-advanced-BM25.ipynb`

BM25 (Best Match 25) is a keyword-based ranking function that improves on TF-IDF by adding term-frequency saturation and document-length normalization.

**Scoring formula:**

$$\text{score}(D, Q) = \sum_{i=1}^{n} \text{IDF}(q_i) \cdot \frac{f(q_i, D) \cdot (k_1 + 1)}{f(q_i, D) + k_1 \cdot \left(1 - b + b \cdot \frac{|D|}{\text{avgdl}}\right)}$$

Parameters: `k1 = 1.5` (saturation constant), `b = 0.75` (length penalty), `avgdl` = corpus average document length.

The notebook builds from a three-document toy corpus up to a full LangChain-integrated retriever, adding stopword removal and stemming along the way.

---

#### `Hybrid Approach_ BM25 + Vector Embeddings.ipynb`

BM25 and dense vectors are complementary: BM25 excels at exact term matches, dense embeddings capture semantic similarity. This notebook implements a two-stage pipeline:

```
Stage 1 — BM25 candidate retrieval
  Fast keyword match → top-20 candidates

Stage 2 — Semantic reranking
  Cosine similarity via sentence-transformers over the top-20 only
  Far cheaper than embedding the entire corpus

Output → top-5 reranked documents
```

Also integrates the Cohere Re-Rank API as a drop-in alternative for Stage 2.

---

### Part 2 — Intermediate: Query Intelligence

---

#### `2-Query-Transformations.ipynb`

Raw user queries are often ambiguous or phrased in a way that doesn't match how relevant documents are written. This notebook reshapes queries before retrieval using four techniques:

**Multi-query** generates 3–5 reformulations of the original query and runs them in parallel. The union of results improves recall.

**RAG Fusion + RRF** fuses ranked result lists from multiple queries using Reciprocal Rank Fusion:

$$\text{RRF}(d) = \sum_{r \in R} \frac{1}{k + r(d)}$$

where `k = 60` is a smoothing constant and `r(d)` is the document's rank in list `r`.

**Step-back prompting** rewrites a specific question into a more general principle, retrieves background context, then re-answers the original question grounded in that context.

**HyDE (Hypothetical Document Embeddings)** prompts the LLM to generate a hypothetical answer document, embeds that document, and uses the embedding as the retrieval query. This bridges the vocabulary gap between questions and source documents.

---

#### `3-Routing-to-Datasources.ipynb`

Not every query belongs to the same data source. This notebook implements two routing strategies.

**Logical routing** uses LLM function calling with structured Pydantic output to classify the query and dispatch it to the correct index — vectorstore, graph database, or SQL database.

**Semantic routing** embeds the incoming query and computes cosine similarity against pre-embedded prompt templates. The query is forwarded to whichever prompt it most closely resembles.

---

#### `4-Query-Construction.ipynb`

Vectorstores expose metadata fields that can be filtered to narrow retrieval. This notebook shows how to auto-generate structured queries from natural-language input:

```
Input:  "Find Python tutorials from 2023 about decorators"

Output: {
  "content": "decorators",
  "metadata_filter": {
    "language": "Python",
    "year": { "$gte": 2023 }
  }
}
```

`SelfQueryRetriever` handles this at inference time. The notebook also covers natural-language-to-SQL and natural-language-to-Cypher (graph database) construction.

---

#### `5-Indexing-to-VectoreDB_s.ipynb`

Standard chunk-and-embed indexing has a fundamental tension: small chunks retrieve precisely but lose context; large chunks preserve context but match imprecisely. This notebook implements three strategies that resolve it.

**Multi-representation indexing** indexes compact LLM-generated summaries for retrieval but returns the full source document at query time. Implements [Dense X Retrieval (arxiv 2312.06648)](https://arxiv.org/abs/2312.06648).

**Parent-document retrieval** stores small child chunks for precise matching but returns their larger parent document.

**RAPTOR** builds a recursive summarization tree so that retrieval can operate at multiple levels of abstraction.

---

#### `6-Retrieval-Mechanisms.ipynb`

Covers the full spectrum of retrieval strategies beyond simple similarity search.

**RAG Fusion** generates multiple query variants, retrieves independently, and merges ranked lists using RRF before passing context to the LLM.

**Re-ranking** is a two-step process: approximate retrieval produces candidates; a cross-encoder or Cohere Re-Rank API reorders them by relevance.

**CRAG retrieval (preview)** evaluates retrieved document relevance and triggers a Tavily web search fallback when retrieved docs fall below a confidence threshold.

---

### Part 3 — Advanced: Agentic and Self-Corrective RAG

---

#### `Self-RAG.ipynb`

*Paper: [Self-RAG (arxiv 2310.11511)](https://arxiv.org/abs/2310.11511)*

The model grades its own retrieved documents and generated output before returning a response. Four reflection tokens govern behavior at runtime:

| Token | Question | Possible Outcomes |
|-------|----------|------------------|
| `Retrieve` | Is retrieval needed? | retrieve / skip |
| `IsREL` | Are retrieved docs relevant? | relevant / irrelevant |
| `IsSUP` | Does the generation follow from the docs? | supported / partial / not supported |
| `IsUSE` | Is the response useful? | scored 1–5 |

Implemented as a LangGraph `StateGraph` with conditional edges that branch based on each grading decision.

---

#### `9-Corrective-RAG.ipynb`

*Paper: [CRAG (arxiv 2401.15884)](https://arxiv.org/pdf/2401.15884)*

CRAG introduces a retrieval evaluator that classifies retrieved documents into one of three confidence bands, each triggering a different downstream action:

```
Retrieved docs → Relevance grader (LLM)
     |
     ├── Correct    → refine document → generate
     ├── Ambiguous  → supplement with Tavily web search → generate
     └── Incorrect  → discard → full web search → generate
```

The self-correction loop prevents the LLM from generating answers grounded in irrelevant documents without surfacing the failure to the user.

---

#### `8-Adaptive-RAG-Agent.ipynb`

*Paper: [Adaptive RAG (arxiv 2403.14403)](https://arxiv.org/pdf/2403.14403)*

Rather than always retrieving, Adaptive RAG first classifies the query to decide how much retrieval effort is warranted:

```
Query classifier (LLM)
     |
     ├── No retrieval   → direct LLM answer
     ├── Single-shot    → one retrieval pass
     └── Iterative      → multiple retrieval + reasoning cycles
```

This avoids the latency and cost of retrieval when the LLM already knows the answer, and scales up effort for multi-hop questions that genuinely require it.

---

#### `7-Agentic-RAG.ipynb`

RAG implemented as a LangGraph agent that decides at each step whether to call the retriever, rather than always retrieving unconditionally.

```
State: { messages: [...] }
         |
    call_model
         |
    [Tool call needed?]
     |             |
    Yes            No
     |             |
  ToolNode    final answer
  (retriever)
     |
  append result to state → call_model
```

Key LangGraph constructs: `StateGraph`, `ToolNode`, `add_conditional_edges`, message-based state that accumulates across agent turns.

---

#### `10-Local-RAG-Agent-with-Llama3.ipynb`

A fully local RAG agent — no API keys, no data leaving the machine. Combines the routing logic of Adaptive RAG, the fallback logic of CRAG, and the self-correction of Self-RAG into a single offline pipeline.

| Component | Tool |
|-----------|------|
| LLM | Ollama + Llama 3 |
| Embeddings | GPT4All / Nomic Embed |
| Vector store | ChromaDB (local) |
| Web fallback | Tavily (optional) |

---

#### `RAGAS_CODE.ipynb`

Production-grade evaluation of RAG pipelines using RAGAS metrics, with retry logic, structured logging, and JSON output suitable for automated CI pipelines.

| Metric | Measures |
|--------|----------|
| Faithfulness | Is every claim in the answer traceable to the retrieved context? |
| Answer Relevancy | Does the answer actually address what was asked? |
| Context Precision | Of the retrieved chunks, what fraction were genuinely useful? |
| Context Recall | Did retrieval surface all the information needed to answer correctly? |

---

## Setup

### Installation

```bash
# Core RAG stack
pip install langchain langchain_community langchain-openai \
            langchainhub chromadb tiktoken langchain_groq langgraph

# Retrieval
pip install rank-bm25 sentence-transformers cohere

# Evaluation
pip install ragas

# Local models (notebook 14 only)
pip install gpt4all langchain-nomic
ollama pull llama3
```

### Environment Variables

```bash
export GROQ_API_KEY="your-groq-key"
export OPENAI_API_KEY="your-openai-key"        # embeddings
export LANGSMITH_API_KEY="your-langsmith-key"   # optional tracing
export TAVILY_API_KEY="your-tavily-key"         # web search fallback
```

### LangSmith Tracing (optional)

```python
import os
os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_PROJECT"] = "rag-project"
```

---

## Suggested Learning Paths

**Complete Beginner** (2–3 days)
```
Text Splitting → Intro to RAG → BM25 Basic → Hybrid BM25 + Vector
```

**Intermediate Practitioner** (1 week)
```
Query Transformations → Routing → Query Construction → Indexing → Retrieval Mechanisms
```

**Advanced / Research-Oriented**
```
Self-RAG → Corrective RAG → Adaptive RAG → Agentic RAG → Local RAG → RAGAS
```

**Production Focus**
```
Hybrid Retrieval → RAGAS Evaluation → Agentic RAG → Local RAG Agent
```

---

## Research Papers Referenced

| Paper | Notebook | arXiv |
|-------|----------|-------|
| Self-RAG: Learning to Retrieve, Generate, and Critique | `Self-RAG.ipynb` | [2310.11511](https://arxiv.org/abs/2310.11511) |
| Corrective Retrieval Augmented Generation | `9-Corrective-RAG.ipynb` | [2401.15884](https://arxiv.org/pdf/2401.15884) |
| Adaptive-RAG: Learning to Adapt Retrieval-Augmented LLMs | `8-Adaptive-RAG-Agent.ipynb` | [2403.14403](https://arxiv.org/pdf/2403.14403) |
| Dense X Retrieval | `5-Indexing-to-VectoreDB_s.ipynb` | [2312.06648](https://arxiv.org/abs/2312.06648) |

---

## Key Concepts Glossary

| Term | Definition |
|------|-----------|
| Chunking | Splitting documents into smaller pieces for embedding and retrieval |
| Embedding | Dense vector representation of text, capturing semantic meaning |
| BM25 | Sparse keyword-based ranking with TF-IDF saturation and length normalization |
| RRF | Reciprocal Rank Fusion — merges multiple ranked retrieval lists into one |
| HyDE | Hypothetical Document Embeddings — embeds a generated answer as the retrieval query |
| CRAG | Corrective RAG — grades retrieved docs and falls back to web search when needed |
| Self-RAG | Self-reflective RAG — grades its own retrieval and generation quality at runtime |
| Adaptive RAG | Routes queries to no/single/iterative retrieval based on query complexity |
| LangGraph | Graph-based framework for building multi-step, stateful agentic workflows |
| RAGAS | RAG Assessment framework — automated evaluation of faithfulness and relevancy |

---

## Contributing

Notebooks are designed to build on each other sequentially. When adding a new notebook:

- Follow the naming convention `N-Topic-Name.ipynb`
- Open with a markdown cell that describes the concept, its motivation, and the paper it implements (if applicable)
- Include the LangSmith tracing setup block
- Structure internal sections as: Environment → Concept → Implementation → Evaluation

---

*Built with LangChain · LangGraph · Groq · ChromaDB · RAGAS*
