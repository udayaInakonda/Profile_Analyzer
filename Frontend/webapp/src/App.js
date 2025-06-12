import React, { useState } from "react";
import './App.css';
import UserMode from "./user-mode/user-mode";
import RecruiterMode from "./recruiter-mode/recruiter-mode";

function App() {
  const [mode, setMode] = useState(null);

  if (mode === "user") {
    return <UserMode onBack={() => setMode(null)} />;
  }
  if (mode === "recruiter") {
    return <RecruiterMode onBack={() => setMode(null)} />;
  }

  return (
    <div className="App">
      <header className="App-header">
        <h2>Profile Analyzer</h2>
        <button onClick={() => setMode("user")}>User Mode</button>
        <button onClick={() => setMode("recruiter")}>Recruiter Mode</button>
      </header>
    </div>
  );
}

export default App;