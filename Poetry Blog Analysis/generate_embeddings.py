# 2generate_embeddings.py

import os
import json
from sentence_transformers import SentenceTransformer

# === Setup ===
CHUNKS_PATH = "output_chunks/normalized_chunks.json"
EMBEDDING_OUTPUT_PATH = "embeddings"
os.makedirs(EMBEDDING_OUTPUT_PATH, exist_ok=True)

# Load preprocessed chunks
with open(CHUNKS_PATH, "r", encoding="utf-8") as f:
    chunks = json.load(f)

# Load model
model = SentenceTransformer("BAAI/bge-base-en-v1.5")

# Generate embeddings
normalized_texts = [chunk["normalized"] for chunk in chunks]
embeddings = model.encode(normalized_texts, convert_to_tensor=False)

# Save with embeddings
output_data = [
    {"chunk": chunk, "embedding": emb.tolist()}
    for chunk, emb in zip(chunks, embeddings)
]

output_file = os.path.join(EMBEDDING_OUTPUT_PATH, "chunked_embeddings.json")
with open(output_file, "w", encoding="utf-8") as f:
    json.dump(output_data, f, indent=2)

print(f"✅ Saved embeddings for {len(embeddings)} chunks to `{output_file}`.")
