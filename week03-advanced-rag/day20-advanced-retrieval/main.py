"""
main.py
The central orchestrator for the Day 20 Advanced Retrieval system.
Wires together BM25, ChromaDB Vector Store, Hybrid Search, RRF, and MMR.
"""

import os
from typing import List, Dict, Any

# Import modular pipeline parts
from bm25 import BM25Retriever
from embeddings import EmbeddingEngine
from vector_store import ChromaVectorStore
from hybrid import HybridRetriever
from rrf import RRFComposer
from mmr import MMRReRanker


def setup_mock_data_files():
    """
    Helper function to ensure sample legal document files exist in the data/ folder.
    """
    os.makedirs("data", exist_ok=True)
    
    files_and_content = {
        "data/nda.txt": (
            "Non-Disclosure Agreement (NDA):\n"
            "This document ensures all core product blueprints, intellectual property, "
            "and technical engineering designs remain strictly confidential. Unauthorized "
            "disclosure of proprietary source code is subject to immediate legal prosecution."
        ),
        "data/employment.txt": (
            "Standard Employment Agreement:\n"
            "The employee agrees to perform engineering duties diligently. Compensation details, "
            "equity vesting structures, and standard benefit enrollment are governed by this contract. "
            "The employee must not disclose proprietary information or roadmaps."
        ),
        "data/contract_termination.txt": (
            "Contract Termination Provisions:\n"
            "This legal clause governs agreement termination procedures. Either party may end "
            "the relationship with 30 days written notification. Immediate termination takes "
            "effect if either party willfully breaches confidentiality clauses."
        )
    }
    
    for file_path, content in files_and_content.items():
        if not os.path.exists(file_path) or os.path.getsize(file_path) == 0:
            with open(file_path, "w", encoding="utf-8") as f:
                f.write(content)
            print(f"[Main Setup] Created default mock sample data file: {file_path}")


def load_documents_from_data_folder() -> tuple:
    """
    Reads text files from data/ directory and parses them into a list of structures.
    """
    documents = []
    doc_ids = []
    metadatas = []
    
    target_dir = "data"
    if not os.path.exists(target_dir):
        return documents, doc_ids, metadatas
        
    for file_name in os.listdir(target_dir):
        if file_name.endswith(".txt"):
            file_path = os.path.join(target_dir, file_name)
            with open(file_path, "r", encoding="utf-8") as f:
                content = f.read().strip()
                
            # Treat each file as an individual chunk for this demonstration
            # In larger apps, you would use a text splitter here
            base_name = os.path.splitext(file_name)[0]
            doc_id = f"id_{base_name}"
            
            documents.append(content)
            doc_ids.append(doc_id)
            metadatas.append({"source": file_name, "category": "legal_framework"})
            
    return documents, doc_ids, metadatas


def main():
    print("==================================================================")
    print("⚡ Starting Day 20 Advanced Retrieval Orchestrator Pipeline ⚡")
    print("==================================================================\n")
    
    # 1. Guarantee data directory has contents to test against
    setup_mock_data_files()
    
    # 2. Extract texts
    documents, doc_ids, metadatas = load_documents_from_data_folder()
    if not documents:
        print("[Error] No text data files found to parse in data/ directory.")
        return
        
    # 3. Initialize all core engine parts
    print("\n--- Initializing Engine Subsystems ---")
    bm25_retriever = BM25Retriever()
    embedding_engine = EmbeddingEngine()
    
    # Clean up any residual test DB directory from previous runs to ensure fresh indexing
    chroma_dir = "chroma_db"
    vector_store = ChromaVectorStore(persist_directory=chroma_dir)
    
    # 4. Build indices
    print("\n--- Document Indexing Phase ---")
    # A. Index into BM25 Lexical Model
    bm25_retriever.index_documents(documents, doc_ids=doc_ids, metadatas=metadatas)
    
    # B. Generate Embeddings & Index into ChromaDB Vector Store
    print("[Embeddings] Computing dense vectors for parsed files...")
    dense_vectors = embedding_engine.get_embeddings(documents)
    vector_store.add_documents(documents, dense_vectors, doc_ids=doc_ids, metadatas=metadatas)
    print("[Success] All items successfully ingested across lexical & dense layers.")

    # 5. Core Search Execution Sandbox
    query = "termination notification due to confidentiality leak"
    print(f"\n==================================================================")
    print(f"🔍 TARGET EVALUATION QUERY: '{query}'")
    print(f"==================================================================")
    
    # --- STRATEGY A: LINEAR HYBRID SEARCH ---
    print("\n[Strategy A] Executing Weighted Linear Hybrid Search...")
    hybrid_orchestrator = HybridRetriever(bm25_retriever, embedding_engine, vector_store)
    # Give a balanced 50/50 split between keywords and meaning vector matches
    hybrid_hits = hybrid_orchestrator.retrieve(query, top_k=2, dense_weight=0.5)
    hybrid_orchestrator.print_results(hybrid_hits, title="Normalized Linear Hybrid Output")

    # --- STRATEGY B: RECIPROCAL RANK FUSION (RRF) ---
    print("\n[Strategy B] Executing Reciprocal Rank Fusion (RRF)...")
    # Retrieve raw list pools independently first
    bm25_raw_run = bm25_retriever.retrieve(query, top_k=3, filter_zeros=False, normalize=False)
    query_vector = embedding_engine.get_query_embedding(query)
    dense_raw_run = vector_store.retrieve(query_vector, top_k=3, normalize=False)
    
    rrf_composer = RRFComposer(k=60)
    rrf_hits = rrf_composer.fuse([bm25_raw_run, dense_raw_run], top_k=2)
    rrf_composer.print_results(rrf_hits, title="Rank-Based RRF Output")

    # --- STRATEGY C: MAXIMAL MARGINAL RELEVANCE (MMR DIVERSITY) ---
    print("\n[Strategy C] Executing Dense Vector Search + MMR Diversity Re-ranking...")
    # Fetch a wider net of candidates first from our vector database
    candidate_hits = vector_store.retrieve(query_vector, top_k=3, normalize=False)
    
    # Extract corresponding pre-computed vectors for the candidates returned
    # In a production app, you might fetch these directly from the DB metadata or cache
    candidate_vectors = []
    for hit in candidate_hits:
        idx = doc_ids.index(hit["id"])
        candidate_vectors.append(dense_vectors[idx])
        
    # Initialize re-ranker with high diversity focus (lambda=0.40)
    mmr_reranker = MMRReRanker(lambda_mult=0.40)
    mmr_hits = mmr_reranker.re_rank(
        query_embedding=query_vector,
        candidates=candidate_hits,
        candidate_embeddings=candidate_vectors,
        top_k=2
    )
    
    print("\n======== MMR Re-ranked Diversified Search (Total Output: 2) ========")
    for item in mmr_hits:
        print(f"MMR Rank {item['_mmr_rank']} | Original Vector Sim Score: {item['score']:.4f} | ID: {item['id']}")
        print(f"Snippet: {item['document'][:90]}...")
        print("-" * 65)

    print("\n==================================================================")
    print("🏁 Pipeline execution complete! All strategies successfully evaluated.")
    print("==================================================================")


if __name__ == "__main__":
    main()