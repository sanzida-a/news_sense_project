# 🧠 NewsSense - AI News Intelligence Agent

**NewsSense** is a modular multi-agent system built using OpenAI’s assistant framework. It helps users **track trending topics**, **fact-check claims**, and **summarize news articles** in real-time.

---

## 📁 Project Structure

- `agent.py` – Main implementation containing all agents and tools
- `.env` – Environment variable template (use your BASE_URL, API_KEY, MODEL_NAME)
- `requirements.txt` – Python dependencies

---

## 🤖 Agents Overview

1. **Conversation Agent** – Main controller that routes user queries to correct agents
2. **Trending News Agent** – Detects and analyzes trending topics using simulated web search
3. **Fact Checker Agent** – Verifies claims using simulated Retrieval-Augmented Generation (RAG)
4. **News Summarizer Agent** – Condenses news content into 3–5 bullet points

---

## ⚙️ Setup Instructions

1. Clone the repository and navigate into it.

2. Install the required dependencies:
```bash
pip install -r requirements.txt
```

3. Create a `.env` file with the following structure (already provided):
```env
BASE_URL="https://models.github.ai/inference/v1"
API_KEY="your token here"
MODEL_NAME="openai/gpt-4.1-nano"
```

> ⚠️ This project uses mock data. Replace it with real APIs in production.

---

## ▶️ Running the Demo

To test NewsSense with example queries:

```bash
python demo.py
```

This demo includes:

- Fetching trending news by topic
- Fact-checking claims via simulated RAG
- Summarizing news articles into bullet points

---

## ✅ Features Demonstrated

### 📈 Trending News Agent

- Retrieves trending headlines by topic
- Provides brief trend summaries
- Simulates user preferences

### ✅ Fact Checker Agent

- Verifies news claims using RAG logic
- Provides verdict with rationale
- Simulates source checking

### 📝 News Summarizer Agent

- Summarizes long articles
- Returns 3–5 digestible key points

### 💬 Conversational Agent

- Accepts natural language queries
- Determines user intent: trending, verify, summarize
- Routes to the correct specialist agent

### 🧠 NewsSense (Main System)

- Central controller combining all agents
- Clean, modular Python design

---

## 🔍 Observability

- Integrated with **Logfire** to log tool calls, agent routing, and decisions.
- Uses **Pydantic** for schema validation of tool inputs/outputs.

---

## 📌 Notes

- The project uses **dummy tools** for demo purposes.
- Swap simulated logic with real news APIs or RAG pipelines for production use.

---

## 🛠️ Built With

- `openai`, `langchain`, `pydantic`, `logfire`, `python-dotenv`
- Async agent framework and modular design

---

## 🧪 Example User Queries

```text
👤 What's trending in AI today?
→ Meta releases new LLM, Grok update, DeepMind milestone

👤 Is OpenAI partnering with Apple?
→ No official confirmation. Sources suggest ongoing talks.

👤 Summarize this article: [pasted content]
→ Returns 3–5 key bullet points
```
