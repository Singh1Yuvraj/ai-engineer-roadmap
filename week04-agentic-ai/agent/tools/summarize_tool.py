class SummarizeTool:
    name = "summarize"
    description = "Summarize provided legal text or document content."
    parameters = {
        "type": "object",
        "properties": {
            "content": {
                "type": "string",
                "description": "The raw text or document content to summarize."
            }
        },
        "required": ["content"]
    }

    def run(self, content: str) -> str:
        # Avoid summarizing placeholders; summarize the actual content!
        clean_content = content[:150] + "..." if len(content) > 150 else content
        return f"[SUMMARY]: Concise summary of extracted text: '{clean_content}'"