# 📦 Imports
import fitz
import streamlit as st
import nltk
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer, util
from langchain.text_splitter import (
    CharacterTextSplitter,
    RecursiveCharacterTextSplitter,
    NLTKTextSplitter
)
from nltk.tokenize import word_tokenize

nltk.download('punkt')

# 📝 Predefined Queries
queries = [
    {
        "id": 1,
        "query": "What are the key steps involved in the forecasting process using multiple inputs?",
        "description": "This targets the entire forecasting workflow from problem definition to model deployment."
    },
    {
        "id": 2,
        "query": "How is missing data identified and handled in the forecasting pipeline?",
        "description": "Covers MCAR, MAR, MNAR and imputation methods like mean, regression, hot/cold deck."
    },
    {
        "id": 3,
        "query": "Which imputation methods are described, and when should each be used?",
        "description": "Focuses on explanation and use cases of each imputation technique."
    },
    {
        "id": 4,
        "query": "What role does a centralized data warehouse play in the forecasting process?",
        "description": "Describes how historical/internal/external data is stored and integrated."
    },
    {
        "id": 5,
        "query": "How does the forecasting model incorporate real-time data updates for rolling forecasts?",
        "description": "Explores live data integration for perishable/time-sensitive goods."
    }
]
  # keep your existing queries

# 📄 Load PDF file
def load_pdf(file_path, max_pages=5):
    doc = fitz.open(stream=file_path.read(), filetype="pdf")
    text = ""
    for page_num in range(min(max_pages, len(doc))):
        text += doc.load_page(page_num).get_text()
    return text

# 🔍 FAISS Index
def create_faiss_index(embeddings):
    index = faiss.IndexFlatL2(embeddings.shape[1])
    index.add(embeddings.cpu().detach().numpy())
    return index

# 🔎 Retriever
def search_chunks(query, index, chunks, model, k=3):
    query_embedding = model.encode([query])
    distances, indices = index.search(np.array(query_embedding), k)
    return [chunks[i] for i in indices[0]]

# ✨ Prompting Techniques
def chain_of_thought_prompt(query, context):
    return f"Let's reason step-by-step to answer the question:\nQ: {query}\nContext: {context}"

def role_based_prompt(query, context):
    return f"You are a forecasting analyst.\nQ: {query}\nBased on the context below, answer clearly.\nContext: {context}"

def reflection_prompt(query, context):
    return f"Think about how the information below answers the question, and reflect on the reasoning.\nQ: {query}\nContext: {context}\nReflection:"

# 🧪 Evaluation - F1 Score
def compute_f1(true_answer, predicted_answer):
    true_tokens = word_tokenize(true_answer.lower())
    pred_tokens = word_tokenize(predicted_answer.lower())
    common = set(true_tokens) & set(pred_tokens)
    if not common: return 0.0
    precision = len(common) / len(pred_tokens)
    recall = len(common) / len(true_tokens)
    return 2 * (precision * recall) / (precision + recall) if precision + recall != 0 else 0.0

# 🧪 Evaluation - Cosine Similarity
def compute_cosine_similarity(query, context, model):
    embeddings = model.encode([query, context], convert_to_tensor=True)
    return float(util.pytorch_cos_sim(embeddings[0], embeddings[1]))

# 🚀 Streamlit UI
st.set_page_config(page_title="RAG", layout="wide")
st.title("🔍 RAG with FAISS: Chunking, Prompting & Evaluation")

uploaded_file = st.file_uploader("📤 Upload PDF", type="pdf")

if uploaded_file:
    text = load_pdf(uploaded_file)
    st.subheader("📄 Text Preview")
    st.code(text[:1000])

    # 🔨 Chunking
    fixed_chunks = CharacterTextSplitter(chunk_size=500, chunk_overlap=50).split_text(text)
    recursive_chunks = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50).split_text(text)

    st.write(f"🔹 Fixed Chunks: {len(fixed_chunks)}")
    st.write(f"🔹 Recursive Chunks: {len(recursive_chunks)}")

    model = SentenceTransformer('all-MiniLM-L6-v2')
    embeddings_fixed = model.encode(fixed_chunks, convert_to_tensor=True)
    embeddings_recursive = model.encode(recursive_chunks, convert_to_tensor=True)

    st.success("✅ Embeddings Done")

    index_fixed = create_faiss_index(embeddings_fixed)
    index_recursive = create_faiss_index(embeddings_recursive)

    st.subheader("🔍 Retrieval + Prompt Generation + Evaluation")

    prompt_types = {
        "Chain-of-Thought": chain_of_thought_prompt,
        "Role-based": role_based_prompt,
        "Reflection": reflection_prompt
    }

    eval_type = st.selectbox("📊 Select Evaluation Metric", ["F1 Score", "Cosine Similarity"])

    for q in queries:
        st.markdown(f"### 🔹 Query {q['id']}: {q['query']}")
        st.caption(q['description'])

        for method_name, chunks, index in [
            ("Fixed-size", fixed_chunks, index_fixed),
            ("Recursive", recursive_chunks, index_recursive)
        ]:
            with st.expander(f"📘 Retrieved: {method_name} Chunks"):
                retrieved = search_chunks(q["query"], index, chunks, model)
                for i, context in enumerate(retrieved):
                    prompt_func = list(prompt_types.values())[i % 3]  # Rotate prompts
                    prompt = prompt_func(q["query"], context)

                    st.text_area(f"🔹 {method_name} Result {i+1} ({list(prompt_types.keys())[i % 3]})", prompt,
                                 key=f"{method_name}_{q['id']}_{i}")

                    # Evaluate
                    if eval_type == "F1 Score":
                        score = compute_f1(q["query"], context)
                    else:
                        score = compute_cosine_similarity(q["query"], context, model)
                    st.write(f"📊 **{eval_type}**: {score:.2f}")
