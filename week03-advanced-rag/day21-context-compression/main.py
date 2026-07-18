"""
main.py
Day 21 Context Compression Orchestrator.
Processes raw legal contract text files, maps parent-child splits, 
and hosts a live interactive CLI retrieval optimization sandbox.
"""

import os
from embeddings import EmbeddingEngine
from vector_store import ChromaVectorStore
from parent_child import ParentChildRetriever
from retrieval import AdvancedCompressedRetriever
from compare import ContextMetricsEvaluator
from chunkers import LegalHierarchicalChunker


def load_data_from_folder() -> str:
    """Reads all target contract txt files from data/ directory."""
    corpus_blocks = []
    target_dir = "data"
    expected_files = ["nda.txt", "employment.txt", "contract_termination.txt"]
    
    if os.path.exists(target_dir):
        for file_name in expected_files:
            file_path = os.path.join(target_dir, file_name)
            if os.path.exists(file_path):
                print(f"[Main Ingestion] Reading file: {file_name} ({os.path.getsize(file_path)} bytes)")
                with open(file_path, "r", encoding="utf-8") as f:
                    corpus_blocks.append(f.read().strip())
            else:
                print(f"[Warning] Expected file missing: {file_path}")
                    
    return "\n\n".join(corpus_blocks)


def main():
    print("==================================================================")
    print("⚡ Day 21 Interactive Context Compression Engine ⚡")
    print("==================================================================\n")

    # 1. Initialize core infrastructure units
    encoder = EmbeddingEngine()
    store = ChromaVectorStore(persist_directory="chroma_db")
    pc_manager = ParentChildRetriever()
    
    # Sharp chunking constraints optimal for finding precise legal boundaries
    chunker = LegalHierarchicalChunker(child_size=35, overlap=8) 
    evaluator = ContextMetricsEvaluator()
    retriever = AdvancedCompressedRetriever(store, encoder, pc_manager)

    # 2. Extract texts from data files
    raw_document_text = load_data_from_folder()
    if not raw_document_text:
        print("[Error] No text data files found in data/ folder.")
        return

    print("\n[Ingestion] Splitting text corpus into linked Hierarchical Parent-Child items...")
    data_packages = chunker.create_parent_child_packages(raw_document_text, "legal_contract.txt")

    # 3. Structural Index Processing Loop
    print(f"[Ingestion] Indexing {len(data_packages)} structural text packages...")
    for p in data_packages:
        vec = encoder.get_embeddings([p["child_text"]])[0]
        store.add_documents([p["child_text"]], [vec], [p["child_id"]], [p["metadata"]])
        pc_manager.register_relationship(
            child_id=p["child_id"], 
            parent_id=p["parent_id"], 
            parent_text=p["parent_text"], 
            neighbors=p["neighbors"], 
            metadata=p["metadata"]
        )
    print("[Success] All indices constructed safely. Entering interactive loop.")
    print("==================================================================")

    # 4. Interactive CLI Evaluation Suite
    while True:
        try:
            query = input("\nEnter your legal query (type 'exit' to quit): ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\nExiting...")
            break

        if query.lower() in ["exit", "quit", "q"]:
            print("Exiting...")
            break

        if not query:
            print("Please enter a valid search string.")
            continue

        print(f"\n🔍 Searching for: '{query}'")
        
        # Run Expanded Parent Context mode (Full raw paragraph chunks)
        expanded_parent_hits = retriever.search(query, mode="parent_expanded", top_k=2)
        
        # Run Compressed Context mode (Stripping out non-matching sentence structures)
        compressed_hits = retriever.search(query, mode="compressed", top_k=2)

        if not expanded_parent_hits:
            print("❌ No matching text sections found for this specific query.")
            continue

        # 5. Evaluate Metrics Comparison
        metrics = evaluator.analyze_compression_efficiency(expanded_parent_hits, compressed_hits)
        evaluator.display_report(metrics, title="Day 21 Production Context Compression Audit")

        if compressed_hits:
            print("\n🔎 Optimized Compressed Context Window Payload:")
            print("-" * 70)
            print(compressed_hits[0]["document"])
            print("-" * 70)


if __name__ == "__main__":
    main()