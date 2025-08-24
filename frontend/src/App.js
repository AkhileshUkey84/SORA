import React, { useState, useEffect } from "react";
import AuthPage from "./components/AuthPage";
import Dashboard from "./components/Dashboard";

function App() {
  const [currentUser, setCurrentUser] = useState(null);

  useEffect(() => {
    const savedUser = localStorage.getItem("currentUser");
    if (savedUser) {
      setCurrentUser(JSON.parse(savedUser));
    }
  }, []);

  return (
    <>
      {!currentUser ? (
        <AuthPage setCurrentUser={setCurrentUser} />
      ) : (
        <Dashboard currentUser={currentUser} setCurrentUser={setCurrentUser} />
      )}
    </>
  );
}

export default App;
