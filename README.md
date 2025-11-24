📌 README — RAG Pipeline with ChromaDB, BM25, MMR & Hybrid Retrieval

This repository contains a complete Retrieval-Augmented Generation (RAG) pipeline built using:

LangChain

ChromaDB vector store

SentenceTransformer embeddings

BM25 retriever

MMR retriever

Hybrid retriever

Groq LLM (llama-3.1-8b-instant)

Post-processing tools (summarization, filtering, faithfulness scoring)

The pipeline supports PDF ingestion, chunking, embedding, vector storage, retrieval, hybrid search, and generation.

🚀 1. Project Overview

This RAG pipeline allows you to:

✔ Load multiple PDFs
✔ Split text using different chunking strategies
✔ Generate sentence embeddings
✔ Store embeddings in ChromaDB
✔ Retrieve documents using:

Dense vector search

BM25 (term-based search)

Maximal Marginal Relevance (MMR)

Hybrid BM25 + Embedding retrieval

✔ Feed retrieved context into Groq LLM
✔ Generate final answers
✔ Calculate faithfulness score
✔ Summarize retrieved passages
✔ Filter context by keywords

This creates a full end-to-end NLP search + generation system.

📦 2. Requirements

All required packages are listed in requirements.txt.

langchain>=0.2.0
langchain-community>=0.2.0
langchain-text-splitters
pypdf
pymupdf
sentence-transformers
faiss-cpu
chromadb
langchain-groq
python-dotenv
typesense
langchain-openai
langgraph
PyPDF2
rank_bm25

✔ Install requirements
pip install --upgrade pip
pip install -r requirements.txt

📁 3. Project Structure
/your-project
│
├── requirements.txt
├── data/
│   ├── pdfs/               # input PDFs
│   └── vector_store/       # persistent ChromaDB storage
│
├── src/
│   ├── ingestion.py        # PDF ingestion + chunking
│   ├── embeddings.py       # embedding manager
│   ├── vectorstore.py      # Chroma vector store wrapper
│   ├── retrievers.py       # vector, BM25, MMR, hybrid retrieval methods
│   ├── rag_pipeline.py     # RAG + Groq model + faithfulness
│   └── postprocessing.py   # summarization, filtering, query expansion
│
└── README.md

📥 4. Data Ingestion
Load all PDFs from a directory

The function:

process_all_pdfs(pdf_directory)


✔ loads every PDF
✔ extracts pages
✔ attaches metadata
✔ returns LangChain Document objects

📑 5. Chunking Strategies

You may chunk by:

Page

Paragraph

Sentence

Title-based

Characters

Tokens

Overlapping chunks

Example:

chunk_documents(documents, strategy="sentence")

🧠 6. Embedding Manager

A wrapper over SentenceTransformer:

Loads model (all-MiniLM-L6-v2)

Generates embeddings for text

Returns embedding dimension

Usage:

embedding_manager = EmbeddingManager()
embeddings = embedding_manager.generate_embeddings(text_list)

📚 7. ChromaDB Vector Store

The VectorStore class:

Creates persistent folder

Loads/creates ChromaDB collection

Adds documents + embeddings

Example:

vector_store = VectorStore()
vector_store.add_documents(chunks, embeddings)

🔍 8. Retrieval Pipelines
1️⃣ Dense Vector Retrieval (Chroma)

Class: RAGRetriever

retriever.retrieve(query, top_k=5)

2️⃣ BM25 Retriever (no NLTK)

Class: BM25Retriever

3️⃣ Maximal Marginal Relevance (MMR)

Class: MMRRetriever

4️⃣ Hybrid Retriever (BM25 + Dense)

Class: HybridRetriever

Each returns:

{
  "id": "...",
  "content": "...",
  "metadata": {...},
  "similarity_score": ...,
  "rank": 1
}

🤖 9. Groq LLM Generation

Using llama-3.1-8b-instant through ChatGroq:

llm = ChatGroq(groq_api_key, model_name="llama-3.1-8b-instant")

RAG pipeline:
answer, score = rag_simple_with_faithfulness(query, retriever, llm, embedding_manager)

📏 10. Faithfulness Scoring

We compute cosine similarity between:

embedding(context)

embedding(answer)

Score → 0 to 1
Higher = more faithful to the context.

🛠 11. Post Processing
✔ Summarization (BART)
summarize_passages(retrieved_docs)

✔ Keyword Filtering
filter_relevant_context(retrieved_docs, ["keyword1", "keyword2"])

✔ Query Expansion

For retrieval improvement.

🧪 12. How to Run End-to-End
# Step 1: Load PDFs
docs = process_all_pdfs("data/pdfs")

# Step 2: Chunk
chunks = chunk_documents(docs, strategy="sentence")

# Step 3: Embeddings
embeddings = embedding_manager.generate_embeddings([c.page_content for c in chunks])

# Step 4: Store in Chroma
vector_store.add_documents(chunks, embeddings)

# Step 5: Initialize retriever
retriever = RAGRetriever(vector_store, embedding_manager)

# Step 6: Ask question
answer, score = rag_simple_with_faithfulness("What is the purpose of the document?", retriever, llm, embedding_manager)

📌 13. Environment Variables

Create .env file:

GROQ_API_KEY=your_key_here

🤝 14. Contributing

Pull Requests are welcome — improvements in retrieval, generation, or post-processing are appreciated.

📜 15. License

This project is open-source; feel free to modify and extend.
