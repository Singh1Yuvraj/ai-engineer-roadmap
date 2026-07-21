"""
agent/tools/search_tool.py
Search Tool that scans real text files inside data/.
"""

import os
from typing import Dict, Any, List


class LegalSearchTool:
    def __init__(self, data_dir: str = "data"):
        self.name = "search"
        self.description = "Searches legal documents in data/ for relevant clauses and terms."
        self.data_dir = data_dir

    def run(self, query: str) -> List[Dict[str, Any]]:
        """Executes keyword matching over files in data/."""
        results = []
        if not os.path.exists(self.data_dir):
            return [{"file": "error", "document": "Data folder not found.", "score": 0.0}]

        keywords = [w.lower() for w in query.split() if len(w) > 3]

        for file_name in os.listdir(self.data_dir):
            if file_name.endswith(".txt"):
                file_path = os.path.join(self.data_dir, file_name)
                with open(file_path, "r", encoding="utf-8") as f:
                    lines = f.readlines()

                matching_lines = []
                for line in lines:
                    line_str = line.strip()
                    if not line_str:
                        continue
                    if any(kw in line_str.lower() for kw in keywords):
                        matching_lines.append(line_str)

                if matching_lines:
                    results.append({
                        "file": file_name,
                        "document": " ".join(matching_lines[:3]),
                        "score": 0.90
                    })

        if not results:
            results.append({
                "file": "none",
                "document": f"No matching lines found across files in data/ for query: '{query}'",
                "score": 0.0
            })

        return results