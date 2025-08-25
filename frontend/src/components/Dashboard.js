import React, { useState } from "react";
import UploadSection from "./UploadSection";
import QuerySection from "./QuerySection";
import HistorySection from "./HistorySection";

function Dashboard({ currentUser, setCurrentUser }) {
  const [section, setSection] = useState("upload");
  const [dataset, setDataset] = useState(null);
  const [queries, setQueries] = useState([]);

  const addQuery = (query) => {
    setQueries([query, ...queries]);
  };

  const handleLogout = () => {
    localStorage.removeItem("currentUser");
    setCurrentUser(null);
  };

  return (
    <div id="dashboard-page" className="page active">
      <div className="dashboard-layout">
        {/* Sidebar */}
        <aside className="sidebar">
          <div className="sidebar-header">
            <h1>AI Data Analyst</h1>
          </div>
          <nav className="sidebar-nav">
            <button
              className={`nav-item ${section === "upload" ? "active" : ""}`}
              onClick={() => setSection("upload")}
            >
              Upload Dataset
            </button>
            <button
              className={`nav-item ${section === "query" ? "active" : ""}`}
              onClick={() => {
                if (dataset) setSection("query");
                else alert("Please upload a dataset first.");
              }}
            >
              Ask Question
            </button>
            <button
              className={`nav-item ${section === "history" ? "active" : ""}`}
              onClick={() => setSection("history")}
            >
              View History
            </button>
          </nav>
        </aside>

        {/* Main */}
        <main className="main-content">
          <header className="main-header">
            <h2>
              {section === "upload"
                ? "Dashboard"
                : section === "query"
                ? "Ask Question"
                : "Query History"}
            </h2>
            <div className="header-actions">
              <span className="user-email">{currentUser.email}</span>
              <button className="logout-btn" onClick={handleLogout}>
                Logout
              </button>
            </div>
          </header>

          <div className="content-area">
            {section === "upload" && (
              <UploadSection
                setDataset={setDataset}
                onUploadComplete={() => setSection("query")}
              />
            )}
            {section === "query" && (
              <QuerySection dataset={dataset} addQuery={addQuery} />
            )}
            {section === "history" && <HistorySection queries={queries} />}
          </div>
        </main>
      </div>
    </div>
  );
}

export default Dashboard;
