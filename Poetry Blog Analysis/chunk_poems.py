#1 chunk_poems.py

import os
import json
import nltk
import re
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.corpus import wordnet

# === Setup ===
nltk.download("punkt")
nltk.download("wordnet")
nltk.download("omw-1.4")

DATA_PATH = "data"
CHUNKS_PATH = "output_chunks"
MIN_SENTENCE_LENGTH = 8

lemmatizer = WordNetLemmatizer()

# === Helpers ===
def clean_text(text):
    text = re.sub(r'\s+', ' ', text)
    text = re.sub(r'[^\x00-\x7F]+', '', text)
    text = re.sub(r'\b(poem|title|author):?\b', '', text, flags=re.I)
    return text.strip()

def synonym_normalize(tokens):
    normalized = []
    for token in tokens:
        synsets = wordnet.synsets(token)
        if synsets:
            lemma = synsets[0].lemmas()[0].name()
            normalized.append(lemma.lower())
        else:
            normalized.append(token.lower())
    return normalized

def lemmatize_text(text):
    tokens = word_tokenize(text.lower())
    clean_tokens = [lemmatizer.lemmatize(token) for token in tokens if token.isalnum()]
    normalized_tokens = synonym_normalize(clean_tokens)
    return " ".join(normalized_tokens)

# === Load Poems & Chunk ===
documents = []
file_list = sorted([f for f in os.listdir(DATA_PATH) if f.endswith(".txt")])

for filename in file_list:
    with open(os.path.join(DATA_PATH, filename), "r", encoding="utf-8") as f:
        documents.append(f.read())

chunks = []
for idx, doc in enumerate(documents):
    cleaned = clean_text(doc)
    sentences = sent_tokenize(cleaned)
    for s_idx, sentence in enumerate(sentences):
        if len(sentence.split()) >= MIN_SENTENCE_LENGTH:
            norm = lemmatize_text(sentence)
            chunks.append({
                "doc_id": f"poem_{idx + 1}",
                "sentence_id": s_idx,
                "text": sentence,
                "normalized": norm
            })

# === Save Chunks ===
os.makedirs(CHUNKS_PATH, exist_ok=True)
chunk_file = os.path.join(CHUNKS_PATH, "normalized_chunks.json")
with open(chunk_file, "w", encoding="utf-8") as f:
    json.dump(chunks, f, indent=2)

print(f"✅ Saved {len(chunks)} chunks to `{chunk_file}`.")
