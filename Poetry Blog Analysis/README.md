
# RAG – Poetry-based Retrieval Augmented Generation System

RAG is a domain-specific Retrieval-Augmented Generation (RAG) pipeline that enables semantic understanding and generation of responses over poetic documents. It leverages language model embeddings, vector search using Weaviate, and a custom retriever-generator architecture to produce context-aware outputs.

---

## 📌 Objectives

- Enable semantic search across poetic content using embeddings.
- Retrieve relevant poetic chunks for a given question using vector similarity.
- Use large language models (LLMs) to generate answers based on retrieved context.
- Evaluate the quality of generated answers using custom metrics.

---

## 🚀 Features

✅ Poetry document chunking and preprocessing  
✅ Embedding generation using LLMs  
✅ Vector storage in Weaviate for semantic retrieval  
✅ Question answering using LLM + context  
✅ Evaluation with quantitative and qualitative analysis  
✅ Modular and extensible architecture

---

## 🗂️ Project Structure

```
teenuRag/
│
├── app.py                     # Main pipeline controller
├── chunk_poems.py             # Splits poems into chunks for embedding
├── generate_embeddings.py     # Generates embeddings for chunks
├── ingest_to_weaviate.py      # Ingests chunks & embeddings into Weaviate
├── retriever_generator.py     # Retrieves and generates answers
├── poetry_rag_core.py         # Core business logic for RAG
├── evaluate_rag.py            # Evaluates answer quality
├── requirements.txt           # Python dependencies
│
├── data/                      # Raw poetry documents
├── embeddings/                # Stored embeddings
├── evaluation/                # Evaluation data and results
├── output_chunks/             # Normalized & preprocessed chunks
├── .env                       # API keys and configuration
```

---

## 📥 Installation

1. **Clone the Repository**

```bash
git clone https://github.com/yourusername/Rag.git
cd Rag
```

2. **Install Dependencies**

```bash
pip install -r requirements.txt
```

3. **Configure Environment Variables**

Create a `.env` file:

```env
OPENAI_API_KEY=your_openai_api_key
WEAVIATE_ENDPOINT=http://localhost:8080
```

---

## 🧪 Workflow

### Step 1: Chunk Documents

```bash
python chunk_poems.py
```

### Step 2: Generate Embeddings

```bash
python generate_embeddings.py
```

### Step 3: Ingest to Weaviate

```bash
python ingest_to_weaviate.py
```

### Step 4: Ask Questions

```bash
python app.py
```

### Step 5: Evaluate the System

```bash
python evaluate_rag.py
```

---

## 📊 Evaluation

Evaluation is done using a set of curated QA pairs and the results are stored in:

- `evaluation/evaluation_data.json`
- `evaluation/evaluation_report.json`

Metrics include:

- Embedding similarity
- Response relevance score
- Exact match / partial match

---

## ⚙️ Tech Stack

- **Python 3.10+**
- **OpenAI API** – for embedding & generation
- **Weaviate** – vector DB for retrieval
- **LangChain / custom RAG logic**
- **dotenv** – for config management

---

## 📄 Sample Use Case

> **Question:** *"What does the poet suggest about hope in doc 3?"*  
> **Response:** *(based on doc 3)* "The poet conveys hope as an internal spark that survives even in the harshest of times..."

---

## 🔓 License

This project is licensed under the MIT License. See the `LICENSE` file for details.

---

## 👩‍💻 Author

**Adlin Teenu X**   
August 2025

Project inspired by the intersection of AI and creative literature. Built as part of an exploratory RAG implementation using poetic content.

---

## 🙌 Acknowledgements

- OpenAI for LLM APIs
- Weaviate team for open-source vector DB
- Community behind RAG and LangChain

---
