---

## 📄 `day23-llm-planning-agent/README.md`

```markdown
# Day 23: LLM Planning Agent

## 📌 Overview
Day 23 replaces the rigid `if/elif` rule-based planner from Day 22 with an **LLM-Driven Planner**. The LLM is restricted strictly to generating a structured JSON execution plan based on user query semantics. The executor remains 100% deterministic Python with zero LLM involvement.

---

## 🏗️ Architecture & Pipeline

```text
User Query ──► [ LLM Planner ] ──► Structured JSON Plan ──► [ JSON Parser ]
                                                                 │
Final Response ◄── [ Response LLM ] ◄── Observation ◄── [ Executor ]