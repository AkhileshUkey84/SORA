import React, { useState } from "react";
import ResultsTable from "./ResultsTable";
import { askGemini } from "../api/gemini";

function QuerySection({ dataset, addQuery }) {
  const [question, setQuestion] = useState("");
  const [loading, setLoading] = useState(false);
  const [result, setResult] = useState(null);

  const handleSubmit = async (e) => {
    e.preventDefault();

    if (!dataset || !dataset.data || dataset.data.length === 0) {
      alert("Please upload a dataset first");
      return;
    }

    setLoading(true);

    try {
      // Limit context to first 50 rows to prevent token overflow
      const limitedData = dataset.data.slice(0, 50);

      const geminiResponse = await askGemini(question, limitedData);

      // Ensure geminiResponse has results array
      const resultsArray = Array.isArray(geminiResponse.results)
        ? geminiResponse.results
        : [];

      const query = {
        id: Date.now(),
        question,
        sql: geminiResponse.generatedSQL || "N/A",
        results: resultsArray,
        status: resultsArray.length > 0 ? "success" : "no-results",
        timestamp: new Date(),
      };

      setResult(query);
      addQuery(query);
    } catch (err) {
      console.error("Gemini query error:", err);
      setResult({ question, results: [], status: "error" });
    } finally {
      setLoading(false);
    }
  };

  return (
    <section className="content-section active">
      <div className="section-header">
        <h1>Ask Questions</h1>
        <p>Ask questions in plain English; AI will generate answers from your CSV.</p>
      </div>

      {dataset && (
        <div className="dataset-info">
          <h3>{dataset.name}</h3>
          <p className="dataset-meta">
            {dataset.rows} rows • {dataset.columns} columns • Uploaded {dataset.uploadDate}
          </p>
        </div>
      )}

      <form className="query-form" onSubmit={handleSubmit}>
        <label>What would you like to know?</label>
        <div className="question-input-group">
          <input
            type="text"
            value={question}
            onChange={(e) => setQuestion(e.target.value)}
            placeholder="e.g., What are the top 5 products by sales?"
          />
          <button type="submit" disabled={loading}>
            {loading ? "Processing..." : "Ask"}
          </button>
        </div>
      </form>

      {result && (
        <div className="results-section">
          <div className="results-header">
            <h3>Results</h3>
            <span className={`status-badge ${result.status}`}>
              {result.status}
            </span>
          </div>

          {result.sql && (
            <div className="sql-section">
              <p className="sql-label">Generated SQL:</p>
              <code>{result.sql}</code>
            </div>
          )}

          <ResultsTable results={result.results} />
        </div>
      )}
    </section>
  );
}

export default QuerySection;
