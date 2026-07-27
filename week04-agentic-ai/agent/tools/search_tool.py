from typing import Any, Dict


class SearchTool:
    name = "search"
    description = "Search documents for specific legal terms, clauses, or general facts."
    parameters = {
        "type": "object",
        "properties": {
            "query": {
                "type": "string",
                "description": "Search phrase or topic to locate across legal contracts."
            }
        },
        "required": ["query"]
    }

    def run(self, query: str) -> str:
        return f"[Search Results]: Found matches for query '{query}' in Section 4.1."