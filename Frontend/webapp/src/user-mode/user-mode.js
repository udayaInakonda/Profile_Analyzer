import React, { useState } from "react";

function UserMode({ onBack }) {
  const [jdFile, setJdFile] = useState(null);
  const [resumeFile, setResumeFile] = useState(null);
  const [score, setScore] = useState(null);
  const [loading, setLoading] = useState(false);
  const [insights, setInsights] = useState(null);
  const [insightsLoading, setInsightsLoading] = useState(false);

  const handleSubmit = async (e) => {
    e.preventDefault();
    if (!jdFile || !resumeFile) {
      alert("Please select both files.");
      return;
    }
    setLoading(true);
    setScore(null);
    const formData = new FormData();
    formData.append("jd_pdf", jdFile);
    formData.append("resume_pdf", resumeFile);

    try {
      const response = await fetch("http://127.0.0.1:8000/user/analyze", {
        method: "POST",
        body: formData,
      });
      const data = await response.json();
      setScore(data.score);
    } catch (error) {
      alert("Error uploading files or connecting to backend.");
    }
    setLoading(false);
  };

  const handleInsights = async (e) => {
    e.preventDefault();
    if (!jdFile || !resumeFile) {
      alert("Please select both files.");
      return;
    }
    setInsightsLoading(true);
    setInsights(null);
    const formData = new FormData();
    formData.append("jd_pdf", jdFile);
    formData.append("resume_pdf", resumeFile);

    try {
      const response = await fetch("http://127.0.0.1:8000/user/insights", {
        method: "POST",
        body: formData,
      });
      const data = await response.json();
      setInsights(data.insights || data); // adapt to your backend response
    } catch (error) {
      alert("Error fetching insights.");
    }
    setInsightsLoading(false);
  };

  return (
    <div>
      <button onClick={onBack}>Back</button>
      <h3>User Mode</h3>
      <form onSubmit={handleSubmit}>
        <div>
          <label>Job Description PDF:&nbsp;</label>
          <input
            type="file"
            accept="application/pdf"
            onChange={(e) => setJdFile(e.target.files[0])}
          />
        </div>
        <div>
          <label>Resume PDF:&nbsp;</label>
          <input
            type="file"
            accept="application/pdf"
            onChange={(e) => setResumeFile(e.target.files[0])}
          />
        </div>
        <button type="submit" disabled={loading}>
          {loading ? "Analyzing..." : "Analyze"}
        </button>
        <button
          type="button"
          disabled={insightsLoading || loading}
          onClick={handleInsights}
          style={{ marginLeft: 10 }}
        >
          {insightsLoading ? "Fetching Insights..." : "Insights"}
        </button>
      </form>
      {score !== null && (
        <div style={{ marginTop: 20 }}>
          <h3>Matching Score: {score}</h3>
        </div>
      )}
      {insights && (
        <div style={{ marginTop: 20 }}>
          <h3>Insights:</h3>
          <pre style={{ whiteSpace: "pre-wrap" }}>
            {typeof insights === "string"
              ? insights
              : JSON.stringify(insights, null, 2)}
          </pre>
        </div>
      )}
    </div>
  );
}

export default UserMode;
