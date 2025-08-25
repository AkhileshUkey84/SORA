import React, { useState } from "react";
import Papa from "papaparse";

function UploadSection({ setDataset, onUploadComplete }) {
  const [progress, setProgress] = useState(0);
  const [uploading, setUploading] = useState(false);

  const handleFile = (file) => {
    if (!file.name.endsWith(".csv")) {
      alert("Please upload a CSV file");
      return;
    }
    if (file.size > 100 * 1024 * 1024) {
      alert("File too large (max 100MB)");
      return;
    }

    setUploading(true);
    let progressVal = 0;

    const reader = new FileReader();
    reader.onload = (e) => {
      const fileContent = e.target.result;

      // Parse CSV content into array of objects
      const parsedData = Papa.parse(fileContent, {
        header: true,       // use first row as keys
        skipEmptyLines: true,
      }).data;

      const interval = setInterval(() => {
        progressVal += 20;
        setProgress(progressVal);

        if (progressVal >= 100) {
          clearInterval(interval);
          setTimeout(() => {
            const dataset = {
              name: file.name,
              rows: parsedData.length,
              columns: parsedData[0] ? Object.keys(parsedData[0]).length : 0,
              uploadDate: new Date().toLocaleDateString(),
              data: parsedData, // structured data ready for querying
            };

            setDataset(dataset);
            setUploading(false);
            setProgress(0);

            // Switch to QuerySection
            onUploadComplete();
          }, 500);
        }
      }, 300);
    };

    reader.readAsText(file);
  };

  return (
    <section className="content-section active">
      <div className="welcome-content">
        <h1>Welcome to AI Data Analyst</h1>
        <p>Upload a CSV dataset and start querying in plain English.</p>
      </div>

      <div className="upload-card">
        <div className="upload-header">
          <h3>Upload Dataset</h3>
          <p>Drag and drop your CSV file or click to browse</p>
        </div>

        <div
          className="dropzone"
          onClick={() => document.getElementById("file-input").click()}
        >
          <input
            type="file"
            id="file-input"
            accept=".csv"
            style={{ display: "none" }}
            onChange={(e) =>
              e.target.files.length > 0 && handleFile(e.target.files[0])
            }
          />
          <div className="dropzone-content">
            <h4>Drop your CSV file here</h4>
            <p>Supports files up to 100MB</p>
            <button type="button" className="browse-btn">
              Browse Files
            </button>
          </div>
        </div>

        {uploading && (
          <div className="progress-section">
            <div className="progress-bar">
              <div
                className="progress-fill"
                style={{ width: `${progress}%` }}
              ></div>
            </div>
            <p className="progress-text">Uploading... {progress}%</p>
          </div>
        )}
      </div>
    </section>
  );
}

export default UploadSection;
