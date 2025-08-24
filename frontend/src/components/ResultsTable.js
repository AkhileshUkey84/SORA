export default function Sidebar() {
  return (
    <aside className="sidebar">
      <div className="sidebar-header">
        <h1>SORA</h1>
        <p>AI Data Assistant</p>
      </div>
      <nav className="sidebar-nav">
        <button className="nav-item active">
          <span className="nav-icon">📂</span> Upload
        </button>
        <button className="nav-item">
          <span className="nav-icon">❓</span> Query
        </button>
        <button className="nav-item">
          <span className="nav-icon">📜</span> History
        </button>
      </nav>
    </aside>
  );
}
