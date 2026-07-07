"""
chunkers.py

A production-grade text chunking library implementing multiple strategies
for Retrieval-Augmented Generation (RAG) and LLM data preparation pipelines.

Strategies Included:
1. Fixed-Size Chunking
2. Recursive Character Chunking
3. Sliding Window Chunking
4. Legal/Section-Based Chunking
"""

import abc
import logging
import re
from typing import List, Dict, Any, Optional, Type, Union

# Configure logging
logger = logging.getLogger("chunkers")
if not logger.handlers:
    handler = logging.StreamHandler()
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)


# =====================================================================
# Base Interface & Utilities
# =====================================================================

class BaseChunker(abc.ABC):
    """Abstract Base Class defining the interface for all chunking strategies."""

    @abc.abstractmethod
    def chunk_text(self, text: str) -> List[str]:
        """
        Splits the input text into a list of string chunks.

        Args:
            text: The raw input string to be chunked.

        Returns:
            A list of text chunks.
        """
        pass

    def _validate_input(self, text: str) -> str:
        """Validates and normalizes input text."""
        if text is None:
            raise ValueError("Input text cannot be None.")
        return text.strip()


def normalize_whitespace(text: str) -> str:
    """Utility to collapse multiple spaces/newlines into clean spacing."""
    return re.sub(r'\s+', ' ', text).strip()


# =====================================================================
# Strategy 1: Fixed-Size Chunker
# =====================================================================

class FixedSizeChunker(BaseChunker):
    """
    Chunks text into strict fixed-size character intervals.
    Ideal for uniform embeddings where semantic boundaries are secondary.
    """
    def __init__(self, chunk_size: int = 500, chunk_overlap: int = 50):
        if chunk_size <= 0:
            raise ValueError("chunk_size must be a positive integer.")
        if chunk_overlap < 0 or chunk_overlap >= chunk_size:
            raise ValueError("chunk_overlap must be non-negative and less than chunk_size.")
        
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

    def chunk_text(self, text: str) -> List[str]:
        text = self._validate_input(text)
        if not text:
            return []

        chunks = []
        stride = self.chunk_size - self.chunk_overlap
        
        for i in range(0, len(text), stride):
            chunk = text[i:i + self.chunk_size]
            chunks.append(chunk)
            if i + self.chunk_size >= len(text):
                break
                
        logger.debug(f"FixedSizeChunker created {len(chunks)} chunks.")
        return chunks


# =====================================================================
# Strategy 2: Recursive Character Chunker
# =====================================================================

class RecursiveCharacterChunker(BaseChunker):
    """
    Attempts to split text hierarchically using a list of separators
    (e.g., paragraphs, sentences, words) to keep semantic contexts intact.
    """
    def __init__(
        self, 
        chunk_size: int = 800, 
        chunk_overlap: int = 100, 
        separators: Optional[List[str]] = None
    ):
        if chunk_size <= 0:
            raise ValueError("chunk_size must be a positive integer.")
        if chunk_overlap < 0 or chunk_overlap >= chunk_size:
            raise ValueError("chunk_overlap must be non-negative and less than chunk_size.")
            
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.separators = separators or ["\n\n", "\n", " ", ""]

    def chunk_text(self, text: str) -> List[str]:
        text = self._validate_input(text)
        if not text:
            return []
        return self._recursive_split(text, self.separators)

    def _recursive_split(self, text: str, separators: List[str]) -> List[str]:
        if len(text) <= self.chunk_size:
            return [text]

        if not separators:
            # Fallback to fixed slicing if no separators remain
            return [text[i:i+self.chunk_size] for i in range(0, len(text), self.chunk_size - self.chunk_overlap)]

        separator = separators[0]
        remaining_separators = separators[1:]

        # Split text by the current separator
        if separator == "":
            splits = list(text)
        else:
            splits = text.split(separator)

        chunks = []
        current_chunk = ""

        for split in splits:
            # Re-add separator spacing if it wasn't an empty string split
            potential_next = current_chunk + (separator if current_chunk else "") + split if separator != "" else current_chunk + split
            
            if len(potential_next) <= self.chunk_size:
                current_chunk = potential_next
            else:
                if current_chunk:
                    chunks.append(current_chunk)
                
                # If a single split item is larger than chunk_size, recurse down using next separator
                if len(split) > self.chunk_size:
                    chunks.extend(self._recursive_split(split, remaining_separators))
                    current_chunk = ""
                else:
                    current_chunk = split

        if current_chunk:
            chunks.append(current_chunk)

        # Merge adjacent chunks if they safely fit within overlap constraints
        return self._merge_with_overlap(chunks)

    def _merge_with_overlap(self, pieces: List[str]) -> List[str]:
        merged_chunks = []
        current_chunk = ""

        for piece in pieces:
            if not current_chunk:
                current_chunk = piece
                continue

            potential_merge = current_chunk + " " + piece
            if len(potential_merge) <= self.chunk_size:
                current_chunk = potential_merge
            else:
                merged_chunks.append(current_chunk)
                # Keep tail end of current chunk for overlap calculations
                overlap_start = max(0, len(current_chunk) - self.chunk_overlap)
                current_chunk = current_chunk[overlap_start:] + " " + piece
                
        if current_chunk:
            merged_chunks.append(current_chunk)
            
        return merged_chunks


# =====================================================================
# Strategy 3: Sliding Window Chunker (Token-ready/Word-based)
# =====================================================================

class SlidingWindowChunker(BaseChunker):
    """
    Chunks text based on word count intervals with strict sliding windows.
    Ensures structural overlap across a continuous linguistic stream.
    """
    def __init__(self, window_size: int = 150, step_size: int = 50):
        if window_size <= 0:
            raise ValueError("window_size must be greater than 0.")
        if step_size <= 0 or step_size > window_size:
            raise ValueError("step_size must be greater than 0 and less than or equal to window_size.")
            
        self.window_size = window_size
        self.step_size = step_size

    def chunk_text(self, text: str) -> List[str]:
        text = self._validate_input(text)
        if not text:
            return []

        words = text.split()
        if len(words) <= self.window_size:
            return [" ".join(words)]

        chunks = []
        for i in range(0, len(words), self.step_size):
            window_words = words[i:i + self.window_size]
            chunks.append(" ".join(window_words))
            if i + self.window_size >= len(words):
                break

        return chunks


# =====================================================================
# Strategy 4: Legal / Section-Based Chunker
# =====================================================================

class LegalSectionChunker(BaseChunker):
    """
    Parses complex structural texts (e.g., Legal Documents, Terms of Service)
    by identifying hierarchical clauses, articles, and section definitions.
    """
    def __init__(self, regex_patterns: Optional[List[str]] = None):
        # Default patterns target "Article I", "Section 2.3", "1.1.1", or "Exhibit A"
        self.patterns = regex_patterns or [
            r'(?i)^\s*(?:article|section|clause|exhibit)\s+[A-Z0-9.\-]+',
            r'^\s*[0-9]+\.[0-9]+(?:\.[0-9]+)*\s+'
        ]
        self.compiled_regex = [re.compile(p) for p in self.patterns]

    def _is_section_header(self, line: str) -> bool:
        return any(regex.match(line) for regex in self.compiled_regex)

    def chunk_text(self, text: str) -> List[str]:
        text = self._validate_input(text)
        if not text:
            return []

        lines = text.splitlines()
        chunks = []
        current_chunk_lines = []

        for line in lines:
            if self._is_section_header(line) and current_chunk_lines:
                # Flush the previous section
                chunks.append("\n".join(current_chunk_lines).strip())
                current_chunk_lines = [line]
            else:
                current_chunk_lines.append(line)

        if current_chunk_lines:
            chunks.append("\n".join(current_chunk_lines).strip())

        return [c for c in chunks if c]


# =====================================================================
# Factory Pattern Engine
# =====================================================================

class ChunkerFactory:
    """Unified factory registration engine for spinning up chunking strategies."""
    
    _registry: Dict[str, Type[BaseChunker]] = {
        "fixed": FixedSizeChunker,
        "recursive": RecursiveCharacterChunker,
        "sliding": SlidingWindowChunker,
        "legal": LegalSectionChunker
    }

    @classmethod
    def register_strategy(cls, name: str, chunker_cls: Type[BaseChunker]) -> None:
        """Allows runtime extension to register custom chunking classes."""
        if not issubclass(chunker_cls, BaseChunker):
            raise TypeError("The chunker class must inherit from BaseChunker.")
        cls._registry[name.lower()] = chunker_cls
        logger.info(f"Successfully registered custom strategy: '{name}'")

    @classmethod
    def create(cls, strategy_name: str, **kwargs: Any) -> BaseChunker:
        """Instantiates and returns the requested chunking strategy."""
        strategy_key = strategy_name.lower()
        if strategy_key not in cls._registry:
            raise ValueError(f"Unknown strategy '{strategy_name}'. Allowed: {list(cls._registry.keys())}")
        
        return cls._registry[strategy_key](**kwargs)


# =====================================================================
# Verification / Runtime Execution Block
# =====================================================================

if __name__ == "__main__":
    print("--- Executing Chunkers Production Verification Run ---\n")

    sample_legal_text = (
        "ARTICLE I: DEFINITIONS\n"
        "The term 'System' refers to the proprietary software platform.\n"
        "The term 'User' refers to any verified entity interacting with the platform.\n"
        "SECTION 1.1: REGISTRATION PROTOCOLS\n"
        "All Users must submit authentic credentials during registration.\n"
        "Failure to do so results in immediate termination of access privileges.\n"
        "SECTION 1.2: COMPLIANCE AND LIABILITY\n"
        "Users agree to abide by regional frameworks. Liability is strictly bounded."
    )

    # 1. Test Legal Strategy
    print("[Testing Legal Section Strategy]")
    legal_chunker = ChunkerFactory.create("legal")
    legal_chunks = legal_chunker.chunk_text(sample_legal_text)
    for i, chunk in enumerate(legal_chunks, 1):
        print(f"Chunk {i}:\n{chunk}\n{'-'*30}")

    # 2. Test Recursive Strategy
    print("\n[Testing Recursive Character Strategy]")
    recursive_chunker = ChunkerFactory.create("recursive", chunk_size=120, chunk_overlap=20)
    recursive_chunks = recursive_chunker.chunk_text(sample_legal_text)
    for i, chunk in enumerate(recursive_chunks[:3], 1):  # Printing first 3 for brevity
        print(f"Chunk {i} (Length {len(chunk)}): {repr(chunk)}")
        
    print(f"\nSuccessfully generated {len(recursive_chunks)} recursive chunks total.")
    print("\n--- Verification Complete. Production Code Ready. ---")