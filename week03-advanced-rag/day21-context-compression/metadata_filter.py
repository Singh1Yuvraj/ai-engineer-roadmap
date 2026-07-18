"""
metadata_filter.py
Structured metadata pre-filtering engine for advanced retrieval pipelines.
"""

from typing import List, Dict, Any, Optional
from datetime import datetime


class MetadataFilter:
    def __init__(self):
        pass

    def filter_by_document(self, candidates: List[Dict[str, Any]], doc_name: str) -> List[Dict[str, Any]]:
        """Filters out items that do not match the target document source name."""
        return [c for c in candidates if c.get("metadata", {}).get("source") == doc_name]

    def filter_by_clause(self, candidates: List[Dict[str, Any]], clause_type: str) -> List[Dict[str, Any]]:
        """Filters chunks by clause definition labels (e.g., 'Indemnification', 'Termination')."""
        return [c for c in candidates if clause_type.lower() in str(c.get("metadata", {}).get("clause", "")).lower()]

    def filter_by_contract(self, candidates: List[Dict[str, Any]], contract_type: str) -> List[Dict[str, Any]]:
        """Filters results based on umbrella contract classifications (e.g., 'NDA', 'MSA')."""
        return [c for c in candidates if contract_type.lower() in str(c.get("metadata", {}).get("contract_type", "")).lower()]

    def filter_by_section(self, candidates: List[Dict[str, Any]], section_num: str) -> List[Dict[str, Any]]:
        """Filters by explicit document indexing notation (e.g., 'Section 4.2')."""
        return [c for c in candidates if c.get("metadata", {}).get("section") == section_num]

    def filter_by_date(
        self, 
        candidates: List[Dict[str, Any]], 
        operator: str, 
        target_date_str: str, 
        date_format: str = "%Y-%m-%d"
    ) -> List[Dict[str, Any]]:
        """
        Filters out documents based on temporal constraints.
        Operators supported: 'before', 'after', 'equal'
        """
        try:
            target_dt = datetime.strptime(target_date_str, date_format)
        except ValueError:
            return candidates # Safe return fallback on bad input formatting

        filtered = []
        for c in candidates:
            doc_date_str = c.get("metadata", {}).get("effective_date")
            if not doc_date_str:
                continue
            try:
                doc_dt = datetime.strptime(str(doc_date_str), date_format)
                if operator == "before" and doc_dt < target_dt:
                    filtered.append(c)
                elif operator == "after" and doc_dt > target_dt:
                    filtered.append(c)
                elif operator == "equal" and doc_dt == target_dt:
                    filtered.append(c)
            except ValueError:
                continue
        return filtered

    def filter_by_risk(self, candidates: List[Dict[str, Any]], max_risk_level: int) -> List[Dict[str, Any]]:
        """Filters out chunks exceeding a numerical risk assessment rating threshold (e.g., 1 to 5)."""
        return [c for c in candidates if int(c.get("metadata", {}).get("risk_score", 0)) <= max_risk_level]

    def combine_filters(self, candidates: List[Dict[str, Any]], filter_criteria: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Executes a composite filtering query over all fields provided.
        Example criteria object layout:
        {
            "contract_type": "NDA",
            "max_risk_level": 3,
            "section": "1.1"
        }
        """
        results = list(candidates)
        
        if "document" in filter_criteria:
            results = self.filter_by_document(results, filter_criteria["document"])
        if "clause" in filter_criteria:
            results = self.filter_by_clause(results, filter_criteria["clause"])
        if "contract_type" in filter_criteria:
            results = self.filter_by_contract(results, filter_criteria["contract_type"])
        if "section" in filter_criteria:
            results = self.filter_by_section(results, filter_criteria["section"])
        if "risk" in filter_criteria:
            results = self.filter_by_risk(results, filter_criteria["risk"])
        if "date_rule" in filter_criteria:
            rule = filter_criteria["date_rule"] # Expects dictionary tracking structure
            results = self.filter_by_date(results, rule.get("op"), rule.get("date"))
            
        return results


if __name__ == "__main__":
    print("[Testing MetadataFilter Isolation]")
    mock_chunks = [
        {"id": "c1", "document": "Text...", "metadata": {"contract_type": "NDA", "risk_score": 2, "section": "1.1"}},
        {"id": "c2", "document": "Text...", "metadata": {"contract_type": "MSA", "risk_score": 5, "section": "4.2"}},
    ]
    f = MetadataFilter()
    filtered = f.combine_filters(mock_chunks, {"contract_type": "NDA", "risk": 3})
    print(f"Filtered down to {len(filtered)} items. Matching ID: {filtered[0]['id']}")