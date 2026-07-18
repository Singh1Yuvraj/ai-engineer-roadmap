"""
compressor.py
Intelligent post-retrieval context compression engine.
"""

import re
from typing import List, Dict, Any
import numpy as np


class LegalContextCompressor:
    def __init__(self):
        pass

    def _split_into_sentences(self, text: str) -> List[str]:
        """Utility pattern splitter separating chunks cleanly on legal text boundaries."""
        sentences = re.split(r'(?<!\w\.\w.)(?<![A-Z][a-z]\.)(?<=\.|\?)\s', text)
        return [s.strip() for s in sentences if s.strip()]

    def sentence_compression(self, text: str, min_len: int = 15) -> str:
        """Removes short boilerplate sentences or operational debris."""
        sentences = self._split_into_sentences(text)
        return " ".join([s for s in sentences if len(s) >= min_len])

    def keyword_compression(self, text: str, keywords: List[str]) -> str:
        """Retains only sentences that contain specific keywords of interest."""
        if not keywords:
            return text
        sentences = self._split_into_sentences(text)
        kw_lower = [kw.lower() for kw in keywords]
        
        matched_sentences = []
        for s in sentences:
            if any(kw in s.lower() for kw in kw_lower):
                matched_sentences.append(s)
                
        return " ".join(matched_sentences) if matched_sentences else text

    def top_sentence_selection(self, text: str, query_keywords: List[str], count: int = 2) -> str:
        """Scores sentences based on basic matching density and slices the top N elements."""
        sentences = self._split_into_sentences(text)
        if len(sentences) <= count:
            return text
            
        scored_sentences = []
        for s in sentences:
            score = sum(1 for kw in query_keywords if kw.lower() in s.lower())
            scored_sentences.append((score, s))
            
        # Sort descending by word matching density
        scored_sentences = sorted(scored_sentences, key=lambda x: x[0], reverse=True)
        top_n = scored_sentences[:count]
        return " ".join([item[1] for item in top_n])

    def similarity_compression(
        self, 
        text: str, 
        query_vector: List[float], 
        embedding_engine_fn: Any, 
        threshold: float = 0.3
    ) -> str:
        """Evaluates sub-sentences directly against query vectors and prunes low-similarity parts."""
        sentences = self._split_into_sentences(text)
        if not sentences:
            return text
            
        sentence_vectors = embedding_engine_fn(sentences)
        q_v = np.array(query_vector)
        
        kept_sentences = []
        for idx, s_vec in enumerate(sentence_vectors):
            s_v = np.array(s_vec)
            norm_q = np.linalg.norm(q_v)
            norm_s = np.linalg.norm(s_v)
            
            sim = float(np.dot(q_v, s_v) / (norm_q * norm_s)) if norm_q and norm_s else 0.0
            if sim >= threshold:
                kept_sentences.append(sentences[idx])
                
        return " ".join(kept_sentences) if kept_sentences else sentences[0]

    def redundant_sentence_removal(self, text: str, embedding_engine_fn: Any, max_similarity: float = 0.85) -> str:
        """Prunes adjacent or internal sentences that carry duplicate structural information."""
        sentences = self._split_into_sentences(text)
        if len(sentences) < 2:
            return text
            
        vectors = [np.array(v) for v in embedding_engine_fn(sentences)]
        unique_sentences = [sentences[0]]
        unique_vectors = [vectors[0]]
        
        for i in range(1, len(sentences)):
            current_v = vectors[i]
            is_redundant = False
            
            for uv in unique_vectors:
                norm_c = np.linalg.norm(current_v)
                norm_u = np.linalg.norm(uv)
                sim = float(np.dot(current_v, uv) / (norm_c * norm_u)) if norm_c and norm_u else 0.0
                
                if sim > max_similarity:
                    is_redundant = True
                    break
            
            if not is_redundant:
                unique_sentences.append(sentences[i])
                unique_vectors.append(current_v)
                
        return " ".join(unique_sentences)

    def context_pruning(self, documents: List[Dict[str, Any]], query_keywords: List[str]) -> List[Dict[str, Any]]:
        """Applies a global layout pipeline to compress every dictionary chunk inside a retrieval run."""
        pruned_list = []
        for doc in documents:
            compressed_text = self.keyword_compression(doc["document"], query_keywords)
            compressed_text = self.sentence_compression(compressed_text)
            
            new_doc = doc.copy()
            new_doc["document"] = compressed_text
            pruned_list.append(new_doc)
        return pruned_list