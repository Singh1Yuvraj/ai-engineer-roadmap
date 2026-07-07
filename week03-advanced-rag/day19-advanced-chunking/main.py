import os
import glob
import sys

# Import components from your existing modules
from chunkers import FixedSizeChunker, RecursiveCharacterChunker, SlidingWindowChunker, LegalSectionChunker
from embeddings import EmbeddingModel
from vector_store import VectorStore
from retrieval import LegalRetriever
from compare import ChunkingEvaluator

def load_documents(data_dir="data"):
    """Loads all text files from the data directory."""
    documents = []
    search_pattern = os.path.join(data_dir, "*.txt")
    file_paths = glob.glob(search_pattern)
    
    if not file_paths:
        print(f"\n[!] Warning: No .txt files found in '{data_dir}'. Please add documents first.")
        return documents
        
    for file_path in file_paths:
        with open(file_path, "r", encoding="utf-8") as file:
            documents.append({
                "name": os.path.basename(file_path),
                "text": file.read()
            })
    return documents

def build_index(chunker, chunker_name, documents):
    """Chunks documents, generates embeddings, and indexes them into Chroma."""
    print("\n-----------------------------------------")
    print(f"Loading Documents... ({len(documents)} documents loaded)")
    
    print("Chunking...")
    all_chunks = []
    all_metadatas = []
    
    for doc in documents:
        chunks = chunker.chunk_text(doc["text"])
        for i, chunk in enumerate(chunks):
            all_chunks.append(chunk)
            all_metadatas.append({
                "source": doc["name"],
                "chunk_id": i,
                "strategy": chunker_name
            })
            
    print(f"{len(all_chunks)} chunks created")
    
    print("Generating Embeddings...")
    embedding_model = EmbeddingModel()
    embeddings = embedding_model.embed_documents(all_chunks)
    
    print("Storing in ChromaDB...")
    collection_name = f"main_{chunker_name.lower()}"
    vector_store = VectorStore(collection_name=collection_name)
    
    # Always reset before adding new documents to prevent stale data
    if hasattr(vector_store, 'reset_collection'):
        vector_store.reset_collection()
        
    vector_store.add_documents(
        documents=all_chunks,
        embeddings=embeddings,
        metadatas=all_metadatas
    )
    
    print("Done")
    print("-----------------------------------------")
    
    retriever = LegalRetriever(vector_store=vector_store, embedding_model=embedding_model)
    return retriever

def interactive_search(retriever):
    """Interactive loop for querying the built retriever."""
    while True:
        print("\nAsk Question")
        query = input("> ").strip()
        
        if not query:
            print("Please enter a valid question.")
            continue
            
        print("\n-----------------------------------------")
        results = retriever.retrieve(query, top_k=3)
        
        if not results:
            print("No relevant results found.")
        else:
            for i, res in enumerate(results, 1):
                # Handle dictionary or object formats generically
                score = res.get("score", "N/A") if isinstance(res, dict) else getattr(res, "score", "N/A")
                metadata = res.get("metadata", {}) if isinstance(res, dict) else getattr(res, "metadata", {})
                text = res.get("text", "") if isinstance(res, dict) else getattr(res, "text", "")
                
                source = metadata.get("source", "Unknown")
                
                print(f"Rank {i}")
                # Score is typically distance in Chroma (smaller is better)
                if isinstance(score, float):
                    print(f"Distance Score : {score:.4f}")
                print(f"Source       : {source}")
                print("-----------------------------------------")
                print(f"{text.strip()}\n")
                print("=========================================")
                
        choice = input("Ask another question? (y/n): ").strip().lower()
        if choice != 'y':
            break

def run_comparison(documents):
    """Executes the comparison script over a set of default queries."""
    evaluator = ChunkingEvaluator()
    
    default_queries = [
        "Can employer terminate employee without notice?",
        "What constitutes a breach of the non-disclosure agreement?",
        "Are there any post-termination obligations for the employee?"
    ]
    
    print("\nStarting evaluation. This might take a moment...\n")
    results = evaluator.run_evaluations(documents, default_queries)
    evaluator.print_report(len(documents), default_queries, results)

def main():
    documents = load_documents("data")
    if not documents:
        sys.exit(1)

    chunker_map = {
        "1": ("Fixed", FixedSizeChunker),
        "2": ("Recursive", RecursiveCharacterChunker),
        "3": ("Sliding Window", SlidingWindowChunker),
        "4": ("Legal Section", LegalSectionChunker)
    }

    while True:
        print("\n=========================================")
        print("DAY 19 - ADVANCED CHUNKING")
        print("=========================================")
        print("1. Fixed")
        print("2. Recursive")
        print("3. Sliding Window")
        print("4. Legal Section")
        print("5. Compare All")
        print("6. Exit")
        
        choice = input("\nChoice: ").strip()
        
        if choice == "6":
            print("\nExiting Day 19 Demo. Goodbye!")
            sys.exit(0)
            
        elif choice == "5":
            run_comparison(documents)
            
        elif choice in chunker_map:
            chunker_name, ChunkerClass = chunker_map[choice]
            print(f"\nSelected: {chunker_name} Chunker")
            
            try:
                chunker_instance = ChunkerClass()
                retriever = build_index(chunker_instance, chunker_name, documents)
                interactive_search(retriever)
            except Exception as e:
                print(f"\n[!] Error initializing or running {chunker_name} Chunker: {e}")
                
        else:
            print("\nInvalid choice. Please select 1-6.")

if __name__ == "__main__":
    main()