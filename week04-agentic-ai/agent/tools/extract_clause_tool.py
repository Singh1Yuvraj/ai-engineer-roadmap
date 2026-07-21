"""
agent/tools/extract_clause_tool.py
Extracts specific target legal clauses dynamically from files in data/.
"""

import os


class ExtractClauseTool:
    def __init__(self, data_dir: str = "data"):
        self.name = "extract_clause"
        self.description = "Extracts specific legal clauses directly from files in data/."
        self.data_dir = data_dir

    def run(self, clause_name: str) -> str:
        """Finds paragraphs or sections matching the requested clause in data/."""
        c_lower = str(clause_name).lower()
        extracted_clauses = []

        if not os.path.exists(self.data_dir):
            return "Data directory missing."

        for file_name in os.listdir(self.data_dir):
            if file_name.endswith(".txt"):
                file_path = os.path.join(self.data_dir, file_name)
                with open(file_path, "r", encoding="utf-8") as f:
                    content = f.read()

                # Split into paragraphs or numbered sections
                sections = content.split("\n\n")
                for sec in sections:
                    if c_lower in sec.lower():
                        extracted_clauses.append(f"[{file_name}]\n{sec.strip()}")

        if extracted_clauses:
            return "\n\n---\n\n".join(extracted_clauses)
        
        return f"No clauses containing '{clause_name}' found in data/ files."