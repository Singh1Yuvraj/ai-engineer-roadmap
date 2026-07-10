"""
bm25.py
Production-grade lexical retrieval module using rank_bm25.
Optimized for predictability, customization, and seamless integration 
into hybrid search architectures.
"""

import os
import re
from typing import List, Dict, Any, Optional, Callable
from rank_bm25 import BM25Okapi


def default_tokenizer(text: str) -> List[str]:
    """
    Standard production preprocessing step. 
    Strips punctuation, handles lowercasing, and extracts words.
    """
    return re.findall(r'\b\w+\b', text.lower())


class BM25Retriever:
    def __init__(self, tokenizer_fn: Optional[Callable[[str], List[str]]] = None):
        """
        Initializes the BM25 Retriever with a configurable tokenizer.
        """
        self.tokenizer = tokenizer_fn or default_tokenizer
        self.bm25: Optional[BM25Okapi] = None
        
        # State storage
        self.doc_ids: List[str] = []
        self.documents: List[str] = []
        self.metadatas: List[Dict[str, Any]] = []
        self.tokenized_corpus: List[List[str]] = []

    def index_documents(
        self, 
        documents: List[str], 
        doc_ids: Optional[List[str]] = None, 
        metadatas: Optional[List[Dict[str, Any]]] = None
    ):
        """
        Preprocesses, tokenizes, and indexes documents into the BM25 model.
        """
        if not documents:
            raise ValueError("Document list cannot be empty.")
            
        self.documents = documents
        
        # Guarantee Document IDs exist
        if doc_ids is None:
            self.doc_ids = [f"doc_{i}" for i in range(len(documents))]
        else:
            if len(doc_ids) != len(documents):
                raise ValueError("Length of doc_ids must match length of documents.")
            self.doc_ids = [str(uid) for uid in doc_ids]

        # Handle Metadata
        if metadatas is None:
            self.metadatas = [{} for _ in documents]
        else:
            if len(metadatas) != len(documents):
                raise ValueError("Length of metadatas must match length of documents.")
            self.metadatas = metadatas

        # Separate preprocessing step & store the tokenized corpus
        self.tokenized_corpus = [self.tokenizer(doc) for doc in self.documents]
        
        # Initialize rank_bm25
        self.bm25 = BM25Okapi(self.tokenized_corpus)

    def retrieve(
        self, 
        query: str, 
        top_k: int = 3, 
        filter_zeros: bool = True,
        normalize: bool = False
    ) -> List[Dict[str, Any]]:
        """
        Retrieves matching documents, sorts by relevance, and formats metadata.
        
        Args:
            query: Raw string keyword search query.
            top_k: Upper bound of document matches to return.
            filter_zeros: If True, completely ignores documents with a BM25 score of 0.
            normalize: If True, scales raw scores linearly between 0.0 and 1.0.
            
        Returns:
            Structured list of matches ideal for downstream hybrid ranking.
        """
        if self.bm25 is None:
            raise RuntimeError("BM25 index must be built via index_documents() before querying.")

        tokenized_query = self.tokenizer(query)
        raw_scores = self.bm25.get_scores(tokenized_query)
        
        results = []
        for idx, score in enumerate(raw_scores):
            # Requirement: Filter out completely irrelevant text chunks
            if filter_zeros and score <= 0.0:
                continue
                
            results.append({
                "id": self.doc_ids[idx],
                "document": self.documents[idx],
                "score": float(score),
                "metadata": self.metadatas[idx]
            })
            
        # Sort matches descending by score
        results = sorted(results, key=lambda x: x["score"], reverse=True)
        results = results[:top_k]
        
        # Requirement: Max/Min MinMax Score Normalization for clean hybrid mixing
        if normalize and results:
            max_score = results[0]["score"]
            # Look at the last element in our retrieved or overall subset
            min_score = results[-1]["score"]
            score_range = max_score - min_score
            
            for res in results:
                if score_range > 0:
                    res["score"] = (res["score"] - min_score) / score_range
                else:
                    res["score"] = 1.0  # Safe fallback if all returned scores are perfectly identical
                    
        return results

    def print_results(self, results: List[Dict[str, Any]], title: str = "BM25 Retrieval"):
        """
        Isolated display interface separating printing from business logic.
        """
        print(f"\n=== {title} (Total Returned: {len(results)}) ===")
        for i, res in enumerate(results):
            print(f"Rank {i+1} | ID: {res['id']} | Score: {res['score']:.4f}")
            print(f"Snippet: {res['document'][:90]}...")
            print(f"Metadata: {res['metadata']}")
            print("-" * 50)


if __name__ == "__main__":
    print("[Testing Production BM25Retriever Isolation]")
    
    # Concrete domain setup (Simulating your Day 20 dataset files)
    sample_docs = [
        "Employment Agreement: The employee shall not disclose proprietary code or product roadmaps.",
        "Contract Termination: This agreement terminates immediately if either party breaches confidentiality.",
        "Non-Disclosure Agreement: Recipient agrees to keep all technical blueprints strictly confidential.",
        "Random completely unrelated document text that should get filtered entirely."
    ]
    sample_ids = ["emp_001", "term_002", "nda_003", "unrelated_004"]
    sample_meta = [
        {"file": "employment.txt"}, {"file": "contract_termination.txt"}, 
        {"file": "nda.txt"}, {"file": "garbage.txt"}
    ]
    
    # 1. Custom Tokenizer implementation test
    def custom_stemming_tokenizer(text: str) -> List[str]:
        # Simple lowercase split + trivial suffix truncation for demonstration
        words = re.findall(r'\b\w+\b', text.lower())
        return [w[:-3] if len(w) > 5 else w for w in words]

    retriever = BM25Retriever(tokenizer_fn=custom_stemming_tokenizer)
    retriever.index_documents(sample_docs, doc_ids=sample_ids, metadatas=sample_meta)
    
    # 2. Testing search retrieval with normalization and zero-filtering enabled
    query_str = "confidential termination agreement"
    hits = retriever.retrieve(query_str, top_k=5, filter_zeros=True, normalize=True)
    
    # 3. Print verification using dedicated printing utility
    retriever.print_results(hits, title="Normalized Hybrid-Ready Lexical Query")