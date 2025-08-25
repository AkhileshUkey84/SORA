import React from "react";

function ResultsTable({ results }) {
  if (!results || results.length === 0) return <p>No results found.</p>;

  const columns = Object.keys(results[0]);

  return (
    <div className="results-table">
      <table>
        <thead>
          <tr>
            {columns.map((col) => (
              <th key={col}>{col}</th>
            ))}
          </tr>
        </thead>
        <tbody>
          {results.map((row, i) => (
            <tr key={i}>
              {columns.map((col) => (
                <td key={col}>{row[col]}</td>
              ))}
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}

export default ResultsTable;
