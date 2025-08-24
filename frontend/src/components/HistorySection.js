import React from "react";

function HistorySection({ queries }) {
  if (queries.length === 0) {
    return (
      <div className="empty-state">
        <h3>No query history yet</h3>
        <p>Start by uploading a dataset and asking your first question</p>
      </div>
    );
  }

  return (
    <section className="content-section active">
      <div className="section-header">
        <div>
          <h1>Query History</h1>
          <p>View your previous queries</p>
        </div>
        <div className="query-count">{queries.length} queries</div>
      </div>

      <div className="history-list">
        {queries.map((q) => (
          <div className="history-item" key={q.id}>
            <div className="history-item-header">
              <h4>{q.question}</h4>
              <span className={`status-badge ${q.status}`}>{q.status}</span>
            </div>
            {q.sql && (
              <div className="history-sql">
                <p>Generated SQL:</p>
                <code>{q.sql}</code>
              </div>
            )}
            <div className="history-footer">
              <span>{q.results?.length || 0} results</span>
            </div>
          </div>
        ))}
      </div>
    </section>
  );
}

export default HistorySection;
