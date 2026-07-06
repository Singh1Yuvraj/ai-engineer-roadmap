import os
import chromadb

from sentence_transformers import SentenceTransformer
from langchain_text_splitters import RecursiveCharacterTextSplitter

DATA_PATH = "data"
CHROMA_PATH = "chroma"
COLLECTION_NAME = "legal_documents"


class LegalDocumentIngestor:

    def __init__(self):

        self.model = SentenceTransformer(
            "all-MiniLM-L6-v2"
        )

        self.client = chromadb.PersistentClient(
            path=CHROMA_PATH
        )

        try:
            self.client.delete_collection(
                COLLECTION_NAME
            )
        except:
            pass

        self.collection = self.client.create_collection(
            COLLECTION_NAME
        )

        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=500,
            chunk_overlap=100
        )

    def load_documents(self):

        documents = []

        for file in os.listdir(DATA_PATH):

            if file.endswith(".txt"):

                path = os.path.join(
                    DATA_PATH,
                    file
                )

                with open(
                    path,
                    "r",
                    encoding="utf-8"
                ) as f:

                    documents.append(
                        {
                            "source": file,
                            "content": f.read()
                        }
                    )

        return documents

    def ingest(self):

        documents = self.load_documents()

        all_chunks = []
        metadatas = []
        ids = []

        counter = 0

        for document in documents:

            chunks = self.splitter.split_text(
                document["content"]
            )

            for chunk in chunks:

                all_chunks.append(chunk)

                metadatas.append(
                    {
                        "source": document["source"]
                    }
                )

                ids.append(str(counter))

                counter += 1

        embeddings = self.model.encode(
            all_chunks
        ).tolist()

        self.collection.add(
            ids=ids,
            documents=all_chunks,
            embeddings=embeddings,
            metadatas=metadatas
        )

        print(
            f"\nStored {len(all_chunks)} chunks."
        )


if __name__ == "__main__":

    ingestor = LegalDocumentIngestor()

    ingestor.ingest()