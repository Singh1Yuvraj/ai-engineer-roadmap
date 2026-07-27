# Day 22: Rule-Based AI Agent

## 📌 Overview
Day 22 introduces the core architectural foundations of AI agents by building a deterministic, rule-based agent. Before introducing non-deterministic LLM planning, this day establishes the strict **Separation of Concerns** between intent routing, tool lookup, and execution.

---

## 🏗️ Architecture & Data Flow

```text
User Query ──► [ Rule-Based Planner ] ──► [ Tool Registry ] ──► [ Executor ] ──► Observation