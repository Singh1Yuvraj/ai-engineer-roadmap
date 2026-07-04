import os
import chromadb

from sentence_transformers import SentenceTransformer

DATA_DIR = "data"
CHROMA_DIR = "chroma_db"

model = SentenceTransformer("all-MiniLM-L6-v2")

client = chromadb.PersistentClient(path=CHROMA_DIR)

collection = client.get_or_create_collection(
    name="legal_documents"
)

for filename in os.listdir(DATA_DIR):

    if not filename.endswith(".txt"):
        continue

    filepath = os.path.join(DATA_DIR, filename)

    with open(filepath, "r", encoding="utf-8") as f:
        text = f.read()

    embedding = model.encode(text).tolist()

    collection.add(
        ids=[filename],
        documents=[text],
        embeddings=[embedding],
        metadatas=[
            {
                "source": filename
            }
        ]
    )

print("Documents ingested successfully")