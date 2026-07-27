from typing import Any, Dict, List


class ConversationMemory:

    def __init__(self, max_turns: int = 10):
        self.history: List[Dict[str, str]] = []
        self.max_turns: int = max_turns

    def add_turn(self, user_query: str, final_response: str) -> None:
        self.history.append({"role": "user", "content": user_query})
        self.history.append({"role": "assistant", "content": final_response})
        # Trim older turns if history exceeds max capacity
        if len(self.history) > self.max_turns * 2:
            self.history = self.history[-(self.max_turns * 2) :]

    def get_context_string(self) -> str:
        if not self.history:
            return "No previous conversation history."
        formatted = []
        for msg in self.history:
            formatted.append(f"{msg['role'].capitalize()}: {msg['content']}")
        return "\n".join(formatted)

    def clear(self) -> None:
        self.history.clear()