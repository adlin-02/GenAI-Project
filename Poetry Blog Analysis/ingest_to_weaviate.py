import os
import json
from dotenv import load_dotenv
import weaviate
from weaviate.auth import AuthApiKey
from weaviate.classes.init import AdditionalConfig
from weaviate.collections.classes.config import Property, DataType

# Load environment variables
load_dotenv()

auth = AuthApiKey(os.getenv("WEAVIATE_API_KEY"))

# ✅ Use non-deprecated method
client = weaviate.connect_to_weaviate_cloud(
    cluster_url=os.getenv("WEAVIATE_URL"),
    auth_credentials=auth,
    additional_config=AdditionalConfig(timeout=(30, 300))
)

collection_name = "Poetry"

# ✅ Check if collection exists (correct for v4.16)
existing_collections = client.collections.list_all()
if collection_name not in existing_collections:
    client.collections.create(
        name=collection_name,
        properties=[
            Property(name="content", data_type=DataType.TEXT),
        ],
        vector_index_config={"distance": "cosine"}
    )
    print(f"✅ Created collection: {collection_name}")
else:
    print(f"ℹ️ Collection `{collection_name}` already exists.")

# ✅ Load chunked embeddings
with open("embeddings/chunked_embeddings.json", "r", encoding="utf-8") as f:
    data = json.load(f)

print(f"📦 Loaded {len(data)} chunks from embeddings file")

# ✅ Upload data
collection = client.collections.get(collection_name)
for item in data:
    text = item["chunk"]["text"]
    vector = item["embedding"]
    collection.data.insert(properties={"content": text}, vector=vector)

print("✅ All chunks uploaded to Weaviate successfully.")
client.close()
