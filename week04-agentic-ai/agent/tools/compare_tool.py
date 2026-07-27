from typing import Any, Dict


class CompareTool:
    name = "compare"
    description = "Compare terms across two legal documents or clauses."
    parameters = {
        "type": "object",
        "properties": {
            "doc_a": {
                "type": "string",
                "description": "First document or clause context."
            },
            "doc_b": {
                "type": "string",
                "description": "Second document or clause context."
            }
        },
        "required": ["doc_a", "doc_b"]
    }

    def run(self, doc_a: str, doc_b: str) -> str:
        return f"[Comparison Matrix]: {doc_a} (30 days notice) vs {doc_b} (14 days notice)."