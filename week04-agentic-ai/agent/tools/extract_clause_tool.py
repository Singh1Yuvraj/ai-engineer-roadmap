from typing import Any, Dict

class ExtractClauseTool:
    name = "extract_clause"
    description = "Extract specific legal clauses (e.g., termination, confidentiality, IP, notice period)."
    parameters = {
        "type": "object",
        "properties": {
            "clause_type": {
                "type": "string",
                "description": "Type of clause (e.g., 'termination', 'confidentiality', 'notice period')."
            }
        },
        "required": ["clause_type"]
    }

    def run(self, clause_type: str) -> str:
        clause_map = {
            "termination": "[TERMINATION CLAUSE]: Either party may terminate this agreement upon 30 days written notice.",
            "confidentiality": "[CONFIDENTIALITY CLAUSE]: Both parties agree to maintain strict confidentiality of proprietary data for 2 years.",
            "ip": "[IP CLAUSE]: All intellectual property created during the term belongs exclusively to the Company.",
            "notice period": "[NOTICE PERIOD CLAUSE]: Written notice must be delivered at least 14 business days prior to modification."
        }
        
        normalized = clause_type.strip().lower()
        if normalized in clause_map:
            return clause_map[normalized]
        return f"[{clause_type.upper()} CLAUSE]: Standard contractual obligation regarding {clause_type} applies."