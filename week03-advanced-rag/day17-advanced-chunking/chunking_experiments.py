import os
import re
import chromadb
from sentence_transformers import SentenceTransformer
from langchain_text_splitters import RecursiveCharacterTextSplitter

DATA_DIR = "data"

model = SentenceTransformer("all-MiniLM-L6-v2")


def load_documents():
    docs = []

    for file in os.listdir(DATA_DIR):
        if file.endswith(".txt"):
            path = os.path.join(DATA_DIR, file)

            with open(path, "r", encoding="utf-8") as f:
                docs.append(
                    {
                        "source": file,
                        "content": f.read()
                    }
                )

    return docs


def fixed_chunk(text, chunk_size):
    chunks = []

    for i in range(0, len(text), chunk_size):
        chunks.append(text[i:i + chunk_size])

    return chunks


def recursive_chunk(text, chunk_size):

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=50
    )

    return splitter.split_text(text)


def legal_chunk(text):

    sections = re.split(r"\n\d+\.", text)

    return [
        s.strip()
        for s in sections
        if s.strip()
    ]


def build_collection(method, chunk_size=None):

    client = chromadb.PersistentClient(path="./chroma")

    collection_name = (
        f"{method}_{chunk_size}"
        if chunk_size
        else method
    )

    try:
        client.delete_collection(collection_name)
    except:
        pass

    collection = client.create_collection(collection_name)

    docs = load_documents()

    all_chunks = []
    ids = []
    metadatas = []

    counter = 0

    for doc in docs:

        text = doc["content"]

        if method == "fixed":
            chunks = fixed_chunk(text, chunk_size)

        elif method == "recursive":
            chunks = recursive_chunk(text, chunk_size)

        elif method == "legal":
            chunks = legal_chunk(text)

        else:
            raise ValueError("Unknown method")

        for chunk in chunks:

            all_chunks.append(chunk)

            ids.append(str(counter))

            metadatas.append(
                {
                    "source": doc["source"]
                }
            )

            counter += 1

    embeddings = model.encode(all_chunks).tolist()

    collection.add(
        ids=ids,
        documents=all_chunks,
        embeddings=embeddings,
        metadatas=metadatas
    )

    print(
        f"{collection_name} -> {len(all_chunks)} chunks"
    )

    return collection


if __name__ == "__main__":

    chunk_sizes = [200, 500, 1000]

    for size in chunk_sizes:
        build_collection("fixed", size)

    for size in chunk_sizes:
        build_collection("recursive", size)

    build_collection("legal")