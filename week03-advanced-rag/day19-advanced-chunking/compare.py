import os
import glob
import time

from chunkers import FixedSizeChunker, RecursiveCharacterChunker, SlidingWindowChunker, LegalSectionChunker
from embeddings import EmbeddingModel
from vector_store import VectorStore
from retrieval import LegalRetriever

class ChunkingEvaluator:
    def __init__(self):
        self.embedding_model = EmbeddingModel()
        
        # Initialize all chunkers
        self.chunkers = {
            "Fixed": FixedSizeChunker(),
            "Recursive": RecursiveCharacterChunker(),
            "Sliding": SlidingWindowChunker(),
            "Legal": LegalSectionChunker()
        }

    def load_documents(self, data_dir="data"):
        """Reads all text documents from the specified directory."""
        documents = []
        search_pattern = os.path.join(data_dir, "*.txt")
        file_paths = glob.glob(search_pattern)
        
        if not file_paths:
            print(f"Warning: No .txt files found in '{data_dir}' directory.")
            
        for file_path in file_paths:
            with open(file_path, "r", encoding="utf-8") as file:
                documents.append({
                    "name": os.path.basename(file_path),
                    "text": file.read()
                })
        return documents

    def run_evaluations(self, documents, queries):
        """Runs evaluation across all chunkers, optimizing retriever reuse and catching exceptions."""
        # all_results[query] = [list of chunker metrics]
        all_results = {q: [] for q in queries}
        
        for name, chunker in self.chunkers.items():
            print(f"Evaluating {name} Chunker...")
            try:
                start_time = time.time()
                all_chunks = []
                all_metadatas = []
                
                # 1. Generate Chunks
                for doc in documents:
                    # FIX: Use chunk_text()
                    chunks = chunker.chunk_text(doc["text"])
                    
                    # FIX: Richer metadata
                    for i, chunk in enumerate(chunks):
                        all_chunks.append(chunk)
                        all_metadatas.append({
                            "source": doc["name"],
                            "chunk_id": i,
                            "strategy": name
                        })
                
                if not all_chunks:
                    print(f"  -> Skipped: No chunks generated for {name}.")
                    continue

                # Calculate size metrics (FIX: Added Min/Max)
                chunk_lengths = [len(c) for c in all_chunks]
                total_chunks = len(all_chunks)
                avg_chunk_size = int(sum(chunk_lengths) / total_chunks)
                min_chunk_size = min(chunk_lengths)
                max_chunk_size = max(chunk_lengths)

                # 2. Generate Embeddings
                embeddings = self.embedding_model.embed_documents(all_chunks)

                # 3. Store in Chroma
                # FIX: Better collection naming
                collection_name = f"chunker_{name.lower()}"
                vector_store = VectorStore(collection_name=collection_name)
                
                # FIX: Clear old data before indexing
                if hasattr(vector_store, 'reset_collection'):
                    vector_store.reset_collection()
                    
                vector_store.add_documents(
                    documents=all_chunks, 
                    embeddings=embeddings, 
                    metadatas=all_metadatas
                )
                
                index_time = time.time() - start_time

                # 4. Ask Queries & Retrieve Results
                # FIX: Initialize retriever once per chunker
                retriever = LegalRetriever(vector_store=vector_store, embedding_model=self.embedding_model)
                
                for query in queries:
                    results = retriever.retrieve(query, top_k=5)

                    # 5. Collect Statistics
                    retrieved_count = len(results) if results else 0
                    
                    if results:
                        first_result = results[0]
                        # Assume distance-based scoring from VectorStore (smaller = better)
                        top_score = first_result.get("score", float('inf')) if isinstance(first_result, dict) else getattr(first_result, "score", float('inf'))
                    else:
                        top_score = float('inf')

                    all_results[query].append({
                        "name": name,
                        "chunks": total_chunks,
                        "min_size": min_chunk_size,
                        "max_size": max_chunk_size,
                        "avg_size": avg_chunk_size,
                        "top_score": top_score,
                        "retrieved": retrieved_count,
                        "index_time": index_time
                    })
                    
            except Exception as e:
                # FIX: Prevent one chunker failure from stopping the whole script
                print(f"  -> Error evaluating {name} Chunker: {e}")
                
        return all_results

    def print_report(self, num_docs, queries, all_results):
        """Prints the formatted comparison report."""
        print("\n===============================================================================")
        print("DAY 19 - ADVANCED CHUNKING COMPARISON")
        print("===============================================================================")
        print(f"Documents Loaded : {num_docs}")
        print(f"Queries Tested   : {len(queries)}")
        print(f"Chunkers         : {len(self.chunkers)}")
        print("===============================================================================\n")

        for query in queries:
            metrics_list = all_results.get(query, [])
            if not metrics_list:
                continue
                
            print("Query:")
            print(f"{query}")
            print("-" * 79)
            # FIX: Expanded table format
            print(f"{'Chunker':<12} | {'Chunks':<8} | {'Min':<6} | {'Max':<6} | {'Avg':<6} | {'Top Score':<10} | {'Time':<8}")
            print("-" * 79)
            
            best_chunker = None
            # FIX: Winner logic - searching for the lowest distance
            best_distance = float('inf') 
            
            for m in metrics_list:
                score_str = f"{m['top_score']:.4f}" if m['top_score'] != float('inf') else "N/A"
                time_str = f"{m['index_time']:.2f}s"
                print(f"{m['name']:<12} | {m['chunks']:<8} | {m['min_size']:<6} | {m['max_size']:<6} | {m['avg_size']:<6} | {score_str:<10} | {time_str:<8}")
                
                # Update best chunker (lowest distance wins)
                if m['top_score'] < best_distance:
                    best_distance = m['top_score']
                    best_chunker = m['name']
                    
            print("-" * 79)
            print("Winner (Lowest Distance)")
            
            winner_names = {
                "Fixed": "Fixed Size Chunker",
                "Recursive": "Recursive Character Chunker",
                "Sliding": "Sliding Window Chunker",
                "Legal": "Legal Section Chunker"
            }
            print(f"{winner_names.get(best_chunker, best_chunker)}\n")
            print("===============================================================================\n")


def main():
    evaluator = ChunkingEvaluator()
    
    # 1. Load Documents
    documents = evaluator.load_documents("data")
    
    if not documents:
        print("Exiting: No documents loaded. Please add text files to the 'data' directory.")
        return

    # 2. Define Queries
    queries = [
        "Can employer terminate employee without notice?",
        "What constitutes a breach of the non-disclosure agreement?",
        "Are there any post-termination obligations for the employee?"
    ]

    # 3. Compare Chunkers
    results = evaluator.run_evaluations(documents, queries)

    # 4. Print Final Comparison Report
    evaluator.print_report(len(documents), queries, results)

if __name__ == "__main__":
    main()