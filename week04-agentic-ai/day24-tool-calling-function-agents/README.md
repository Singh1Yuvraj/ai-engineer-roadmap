


# Day 24: Function Calling & Tool Agents

## 📌 Overview
Day 24 upgrades prompt-based JSON planning to **Schema-Based Tool Calling**. By exposing formal JSON Schemas (`name`, `description`, `parameters`, `required` fields) to the LLM, tool selection is strongly typed and validated before execution. It also introduces a centralized `AgentState` object for runtime variable resolution (`$last`) and observation chaining.

---

## 🏗️ Architecture & Centralized AgentState

```text
               User Query
                   │
                   ▼
       ┌───────────────────────┐
       │     LLM Planner       │ ◄── Reads Tool Schemas
       └───────────┬───────────┘
                   │ Tool Calls with State Bindings (e.g. "$last")
                   ▼
       ┌───────────────────────┐
       │      AgentState       │ ◄── Central Memory: Stores outputs,
       └───────────┬───────────┘     execution status & history
                   │ Resolves Arguments ($last ➔ actual text)
                   ▼
       ┌───────────────────────┐
       │     Plan Executor     │ ◄── Schema Validation & Execution
       └───────────┬───────────┘
                   │ Updates History & Results
                   ▼
       ┌───────────────────────┐
       │     Response LLM      │ ──► Final Synthesized Answer
       └───────────────────────┘