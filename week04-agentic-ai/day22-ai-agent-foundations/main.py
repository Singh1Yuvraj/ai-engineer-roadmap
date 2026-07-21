"""
day22-ai-agent-foundations/main.py
Day 22 Entrypoint: Evaluates rule-based agent architecture.
"""

import sys
from pathlib import Path

# Add project root (week04-agentic-ai/) to Python import search path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(PROJECT_ROOT))

from agent.agent import LegalAgent


def print_agent_trace(response: dict):
    print("\n" + "=" * 50)
    print("        LEGAL AI AGENT")
    print("=" * 50)
    
    print(f"\nUser Query:\n{response['query']}")
    
    print("\n---------------- Planner ----------------\n")
    print("Plan:")
    for step in response["plan"]:
        print(f"{step['step']}. {step['tool'].replace('_', ' ').title()} Tool (Input: {step['input']})")
        
    print("\n---------------- Executor ----------------\n")
    for item in response["history"]:
        if item["role"] == "observation":
            print(f"Running {item['tool'].replace('_', ' ').title()} Tool...")
            print(f"Retrieved:\n{item['output']}\n")
            
    print("---------------- Final Answer ----------------\n")
    print(f"{response['final_answer']}")
    print("=" * 50 + "\n")


def main():
    agent = LegalAgent()

    # Test Query 1: Clause Search
    q1 = "Find the termination clause."
    res1 = agent.run(q1)
    print_agent_trace(res1)

    # Test Query 2: Contract Comparison
    q2 = "Compare NDA and Employment Contract"
    res2 = agent.run(q2)
    print_agent_trace(res2)


if __name__ == "__main__":
    main()