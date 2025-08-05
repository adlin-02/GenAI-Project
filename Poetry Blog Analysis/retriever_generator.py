import os
import requests
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer
import weaviate
from weaviate.auth import AuthApiKey
from weaviate.classes.init import AdditionalConfig

# ✅ Load .env
load_dotenv()

# ✅ Load embedding model
embed_model = SentenceTransformer("BAAI/bge-base-en-v1.5")

# ✅ Few-shot examples
FEW_SHOT_EXAMPLES = [
    {
        "question": "What is the poem saying about hope?",
        "answer": "The poem suggests that even during the darkest moments, there is always hope and a reason to keep going."
    },
    {
        "question": "How does the poem describe life?",
        "answer": "Life is portrayed as a complex journey, full of curves, storms, and moments of clarity and stillness."
    }
]

# ✅ Prompt builder
def build_prompt(context, question):
    examples = "\n".join([f"Q: {ex['question']}\nA: {ex['answer']}" for ex in FEW_SHOT_EXAMPLES])
    return f"""{examples}

Context: {context}
Q: {question}
A:"""

# ✅ Connect to Weaviate
client = weaviate.connect_to_weaviate_cloud(
    cluster_url=os.getenv("WEAVIATE_URL"),
    auth_credentials=AuthApiKey(os.getenv("WEAVIATE_API_KEY")),
    additional_config=AdditionalConfig(timeout=(30, 300))
)
collection = client.collections.get("Poetry")

# ✅ Groq LLM Caller
def call_groq_llm(prompt):
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        raise EnvironmentError("Missing GROQ_API_KEY in .env file")

    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }

    body = {
        "model": "llama3-70b-8192",  # or mixtral-8x7b, gemma-7b
        "messages": [
            {"role": "system", "content": "You're a helpful assistant who interprets poetry."},
            {"role": "user", "content": prompt}
        ],
        "temperature": 0.5
    }

    response = requests.post(
        "https://api.groq.com/openai/v1/chat/completions",
        headers=headers,
        json=body
    )

    if response.status_code != 200:
        raise RuntimeError(f"Groq API Error {response.status_code}: {response.text}")

    result = response.json()
    return result["choices"][0]["message"]["content"].strip()

# ✅ RAG + Groq
def get_answer_from_rag(question):
    query_vector = embed_model.encode(question).tolist()
    
    results = collection.query.near_vector(
        near_vector=query_vector,
        limit=5
    )
    
    if not results.objects:
        return "❌ No relevant context found in the vector database."

    context = "\n".join([f"- {obj.properties['content']}" for obj in results.objects])
    prompt = build_prompt(context, question)
    return call_groq_llm(prompt)

# ✅ Run test
if __name__ == "__main__":
    user_question = "What does the poem say about storms and peace?"
    answer = get_answer_from_rag(user_question)
    print("🧠 Answer:", answer)
