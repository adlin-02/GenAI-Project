import os
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer
import weaviate
from weaviate.auth import AuthApiKey
from weaviate.classes.init import AdditionalConfig
import requests
from nltk.tokenize import sent_tokenize

load_dotenv()

# Load embedding model
embed_model = SentenceTransformer("BAAI/bge-base-en-v1.5")

# Few-shot examples
FEW_SHOT_EXAMPLES = [
  {
    "question": "What is the message in 'Let the birds fly'?",
    "answer": "The poem encourages letting go and supporting independence, even when it's emotionally difficult."
  },
  {
    "question": "What does 'I Am The Forest' symbolize?",
    "answer": "It symbolizes resilience and strength through emotional and natural storms."
  }
]


def build_prompt(context, poem_text, question):
    examples = "\n".join([f"Q: {ex['question']}\nA: {ex['answer']}" for ex in FEW_SHOT_EXAMPLES])
    return f"""{examples}

Context:
{context}

Poem:
{poem_text}

Q: {question}
A: Please provide a concise and specific analysis focusing strictly on the question in 2-3 sentences.
"""

# Connect to Weaviate
client = weaviate.connect_to_weaviate_cloud(
    cluster_url=os.getenv("WEAVIATE_URL"),
    auth_credentials=AuthApiKey(os.getenv("WEAVIATE_API_KEY")),
    additional_config=AdditionalConfig(timeout=(30, 300))
)
collection = client.collections.get("Poetry")

# LLM caller using Groq
def call_groq_llm(prompt):
    api_key = os.getenv("GROQ_API_KEY")
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }
    body = {
        "model": "llama3-70b-8192",
        "messages": [
            
            {"role": "system", "content": "You are a poetic reasoning assistant. Always return concise answers focused ONLY on the user question."},
            {"role": "user", "content": prompt}
        ],
        "temperature": 0.5
    }
    try:
        response = requests.post("https://api.groq.com/openai/v1/chat/completions", headers=headers, json=body)
        response.raise_for_status()
        result = response.json()
        return result["choices"][0]["message"]["content"].strip()
    except Exception as e:
        print(f"⚠️ Groq LLM API error: {e}")
        return "This poem likely emphasizes themes of healing, introspection, or emotional growth."

def postprocess_answer(text):
    # Remove duplicate sentences, limit to 3 sentences max
    sentences = sent_tokenize(text)
    seen = set()
    filtered = []
    for s in sentences:
        s_clean = s.strip().lower()
        if s_clean not in seen and len(filtered) < 3:
            filtered.append(s.strip())
            seen.add(s_clean)
    return " ".join(filtered)

def generate_poetry_analysis(query, poem_text=""):
    combined_input = f"Poem: {poem_text}\nQuestion: {query}"
    query_vector = embed_model.encode(combined_input).tolist()

    results = collection.query.near_vector(
        near_vector=query_vector,
        limit=3
    )

    source_chunks = []
    if not results.objects or len(results.objects) == 0:
        context = ""
    else:
        MAX_CONTEXT_CHARS = 1500
        joined_context = ""
        for obj in results.objects:
            addition = obj.properties["content"] + "\n"
            if len(joined_context) + len(addition) > MAX_CONTEXT_CHARS:
                break
            joined_context += addition
            source_chunks.append(obj.properties["content"])
        context = joined_context.strip()

    prompt = build_prompt(context, poem_text, query)
    raw_answer = call_groq_llm(prompt)
    answer = postprocess_answer(raw_answer)

    return answer, source_chunks
