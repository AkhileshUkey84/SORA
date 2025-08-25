## 🚀 Team SORA  

**Kurukshetra 2025 — Hackathon Project**  

We are **Team SORA**, participating in **Kurukshetra 2025**, under the **AI and Agentic AI** domain.  

---

## 👥 Team Members  
- **Rajiv Golait**  
- **Ojas Gharde**  
- **Akhilesh Ukey**  

---

## 💡 Project Overview  
**SORA** is an end-to-end, AI-powered data analytics assistant that lets any user ask questions in plain English and instantly get accurate, trustworthy insights from their datasets. It combines multi-agent orchestration, schema-aware NL→SQL translation, iterative self-correction, proactive insight generation, and transparent security/lineage to deliver reliable, explainable results.  

---

## 🔎 How It Works  
1. **Dataset Ingestion**  
   - Users upload CSV, Excel, or JSON files via a React web interface. Each upload is saved, its schema auto-detected, and a DuckDB view created for fast in-memory querying.  
   - Metadata—including column names, types, sample values, and PII flags—is stored for downstream use.  

2. **Conversational Context & Ambiguity Handling**  
   - Every question runs through a Memory Agent that tracks session history and resolves pronouns or follow-ups (e.g., “What about Q3?”).  
   - An Ambiguity Detector spots vague terms like “recent” or “top” and triggers clarifying questions (e.g., “Last 7 days, 30 days, or this quarter?”).  

3. **Schema-Aware NL→SQL Translation**  
   - A Schema Intelligence layer builds embeddings of columns/tables, infers join paths, and matches user terms to the right fields.  
   - The SQL Generator Agent crafts SQL via a multi-shot approach:  
     - Pattern-based templates for common queries  
     - Few-shot LLM calls with schema-injected prompts  
     - Grammar-constrained generation for complex requests  
   - If the first result isn’t ideal, an iterative self-correction loop applies pattern fixes or issues further LLM prompts until high confidence is reached.  

4. **Multi-Stage Validation & Security**  
   - Generated SQL is vetted by a Validator Agent: syntax checks, semantic validation (e.g., no SUM on text columns), injection prevention, row limits, and PII masking.  
   - Any modifications (e.g., added LIMIT, removed PII columns) are explained in plain language so users understand the guardrails.  

5. **Execution & Caching**  
   - The Executor Agent runs the safe SQL on DuckDB, applying parameter binding and caching results (e.g., Redis) for repeated queries.  
   - Execution metrics (latency, cost estimate, row counts) are logged and exposed for transparency.  

6. **Enhanced Results & Insights**  
   - An Insight Discovery Agent scans result sets for trends, outliers, segment drivers, and surfaces them proactively.  
   - A Self-Audit Agent automatically runs counterfactual queries (e.g., “Exclude promotions”) to compute a confidence score and validate findings.  
   - The Explainer Agent generates a business-friendly narrative, anchoring each key point to specific data rows or aggregates.  

7. **Provenance & Reporting**  
   - A Lineage Agent attaches a reproducibility receipt—query hash, dataset version, timestamps—so any result can be replayed exactly.  
   - An on-demand Report Generator compiles session history into an executive summary PDF or JSON report.  

8. **Frontend & Developer Experience**  
   - A React UI provides pages for dataset management, schema preview, and a rich query interface that renders SQL, tables, charts (Vega-Lite), narratives, insights, audit details, security explanations, and lineage receipts—all in a single view.  
   - Feature toggles let users enable/disable insights, audits, or multimodal output. A “Judge Mode” loads pre-canned scenarios for demo reliability.  

9. **Observability & Reliability**  
   - Structured logging, Prometheus metrics, health checks, and rate-limiting ensure production-grade stability.  
   - Graceful fallbacks—cached responses or offline video clips—guarantee the demo never breaks.  

---

## ✨ Key Features  
- **NL→SQL with schema intelligence**  
- **Multi-agent pipeline**: Memory, SQL Generator, Validator, Executor, Insight, Self-Audit, Explainer, Lineage, Report  
- **Security-first**: validation, parameter binding, row limits, PII masking  
- **Explainability**: narratives, guardrail explanations, reproducibility receipts  
- **Product-grade**: caching, observability, graceful fallbacks  

---

## 🏗️ Tech Stack  
- **Backend**: Python, FastAPI, DuckDB, Pydantic, Uvicorn  
- **Intelligence**: Multi-agent architecture, LLM prompting (provider via env), embeddings  
- **Storage/Cache**: Local files/`uploads/`, DuckDB; optional Redis for caching  
- **Frontend**: React, Vega-Lite  
- **Ops**: Structured logging, health checks, rate limiting  

---

## ⚙️ Setup  
1. Clone the repository  
   ```bash
   git clone https://github.com/<your-repo>.git
   cd SORA
   ```  
2. Backend (Python)  
   ```bash
   cd backend
   python -m venv .venv
   .venv\\Scripts\\activate    # Windows PowerShell
   pip install -r requirements.txt
   python init_database.py         # optional: prepare demo DuckDB
   ```  
   - Set required LLM/API keys as environment variables if needed (e.g., `OPENAI_API_KEY`, `GEMINI_API_KEY`).  
3. Frontend (React)  
   ```bash
   cd ../frontend
   npm install
   ```  

---

## 🚦 How to Run Locally  
1. Start the backend API  
   ```bash
   cd backend
   .venv\\Scripts\\activate
   uvicorn main:app --host 0.0.0.0 --port 8000 --reload
   ```  
2. Start the frontend  
```bash
   cd frontend
   npm start
   ```  
3. Open the app  
   - Visit `http://localhost:3000` for the UI.  
   - Backend available at `http://localhost:8000`.  

---

## 🧪 Demo Tips  
- Use Upload to add CSV/Excel/JSON datasets.  
- Ask: “Top 5 products by revenue in Q3”.  
- Toggle insights/audits to see proactive findings and confidence.  
- Review generated SQL, tables, charts, narrative, guardrails, lineage.  

---

## 🔐 Security & Transparency  
- SQL validation with semantic checks, safe parameter binding, and row limits.  
- PII masking for sensitive columns where applicable.  
- Human-readable explanations for any guardrail-induced changes.  

---

## 📜 License  
MIT (or as applicable).  


