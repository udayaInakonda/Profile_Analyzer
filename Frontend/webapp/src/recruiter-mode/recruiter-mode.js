import React, { useState } from "react";

function RecruiterMode({ onBack }) {
  const [mode, setMode] = useState(null); // null, "existing", "upload", "search"
  const [jdFile, setJdFile] = useState(null);
  const [resumeFiles, setResumeFiles] = useState([]);
  const [results, setResults] = useState(null);
  const [loading, setLoading] = useState(false);
  const [searchQuery, setSearchQuery] = useState("");
  const [searchTopK, setSearchTopK] = useState(5);
  const [searchResults, setSearchResults] = useState(null);
  const [searchLoading, setSearchLoading] = useState(false);

  const handleExistingSubmit = async (e) => {
    e.preventDefault();
    if (!jdFile) {
      alert("Please select a JD file.");
      return;
    }
    setLoading(true);
    setResults(null);
    const formData = new FormData();
    formData.append("jd_pdf", jdFile);

    try {
      const response = await fetch("http://127.0.0.1:8000/recruiter/analyze_db_bulk", {
        method: "POST",
        body: formData,
      });
      const data = await response.json();
      setResults(data.results);
    } catch (error) {
      alert("Error connecting to backend.");
    }
    setLoading(false);
  };

  const handleUploadSubmit = async (e) => {
    e.preventDefault();
    if (!jdFile || resumeFiles.length === 0) {
      alert("Please select a JD and at least one resume.");
      return;
    }
    setLoading(true);
    setResults(null);
    const formData = new FormData();
    formData.append("jd_pdf", jdFile);
    for (const file of resumeFiles) {
      formData.append("resumes", file);
    }

    try {
      const response = await fetch("http://127.0.0.1:8000/recruiter/analyze_bulk", {
        method: "POST",
        body: formData,
      });
      const data = await response.json();
      setResults(data.results);
    } catch (error) {
      alert("Error uploading files or connecting to backend.");
    }
    setLoading(false);
  };

  const handleSearchExistingResumes = async (e) => {
    e.preventDefault();
    if (!searchQuery.trim()) {
      alert("Please enter a search query.");
      return;
    }
    if (searchTopK < 1 || searchTopK > 10) {
      alert("Please enter a value for Top K between 1 and 10.");
      return;
    }
    setSearchLoading(true);
    setSearchResults(null);
    try {
      const response = await fetch("http://127.0.0.1:8000/recruiter/search_existing_resumes", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ query: searchQuery, top_k: searchTopK })
      });
      const data = await response.json();
      setSearchResults(data.results);
    } catch (error) {
      alert("Error searching resumes.");
    }
    setSearchLoading(false);
  };

  return (
    <div>
      <button onClick={onBack}>Back</button>
      <h3>Recruiter Mode</h3>
      {mode === null && (
        <div>
          <button onClick={() => setMode("existing")}>Select from Existing Candidates</button>
          <button onClick={() => setMode("upload")}>Upload New Resumes</button>
          <button onClick={() => setMode("search")}>Search Existing Resumes</button>
        </div>
      )}

      {mode === "existing" && (
        <form onSubmit={handleExistingSubmit}>
          <div>
            <label>Job Description PDF:&nbsp;</label>
            <input
              type="file"
              accept="application/pdf"
              onChange={(e) => setJdFile(e.target.files[0])}
            />
          </div>
          <button type="submit" disabled={loading}>
            {loading ? "Analyzing..." : "Analyze Existing Candidates"}
          </button>
          <button type="button" onClick={() => { setMode(null); setResults(null); }}>Back</button>
        </form>
      )}

      {mode === "upload" && (
        <form onSubmit={handleUploadSubmit}>
          <div>
            <label>Job Description PDF:&nbsp;</label>
            <input
              type="file"
              accept="application/pdf"
              onChange={(e) => setJdFile(e.target.files[0])}
            />
          </div>
          <div>
            <label>Resume PDFs (multiple):&nbsp;</label>
            <input
              type="file"
              accept="application/pdf"
              multiple
              onChange={(e) => setResumeFiles(Array.from(e.target.files))}
            />
          </div>
          <button type="submit" disabled={loading}>
            {loading ? "Analyzing..." : "Analyze Uploaded Resumes"}
          </button>
          <button type="button" onClick={() => { setMode(null); setResults(null); }}>Back</button>
        </form>
      )}

      {mode === "search" && (
        <form onSubmit={handleSearchExistingResumes}>
          <div>
            <label>Search Query:&nbsp;</label>
            <input
              type="text"
              value={searchQuery}
              onChange={(e) => setSearchQuery(e.target.value)}
              placeholder="e.g. Django, REST API, PostgreSQL"
            />
          </div>
          <div>
            <label>Top K (1-10):&nbsp;</label>
            <input
              type="number"
              min={1}
              max={10}
              value={searchTopK}
              onChange={(e) => setSearchTopK(Number(e.target.value))}
              style={{ width: 60 }}
            />
          </div>
          <button type="submit" disabled={searchLoading}>
            {searchLoading ? "Searching..." : "Search Existing Resumes"}
          </button>
          <button type="button" onClick={() => { setMode(null); setSearchResults(null); }}>Back</button>
        </form>
      )}

      {results && (
        <div style={{ marginTop: 20 }}>
          <h3>Results:</h3>
          <ul>
            {results.map((r, idx) => (
              <li key={idx}>
                {r.resume_name}: {r.score}
                {r.error && <span style={{ color: "red" }}> (Error: {r.error})</span>}
              </li>
            ))}
          </ul>
        </div>
      )}

      {searchResults && (
        <div style={{ marginTop: 20 }}>
          <h3>Search Results:</h3>
          <ul>
            {searchResults.map((r, idx) => (
              <li key={idx}>
                {r.resume_name}: {r.score}
              </li>
            ))}
          </ul>
        </div>
      )}
    </div>
  );
}

export default RecruiterMode;
