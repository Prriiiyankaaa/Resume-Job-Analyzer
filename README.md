# Resume-Job-Analyzer

A Python web application that analyzes resumes against job descriptions to help job seekers and recruiters identify the best match.

---

## Features

- Upload a resume (PDF) and paste a job description
- Extracts and compares key skills and keywords
- Calculates a match/similarity score between the resume and job posting
- Stores analysis results in a local database
- View previously analyzed results via a database viewer
- Clean HTML frontend served through Python

---

## Project Structure

```
Resume-Job-Analyzer/
├── resume_analyzer.py      # Main application — analysis logic and web server
├── view_db.py              # Utility script to inspect stored results
├── tempCodeRunnerFile.py   # Auto-generated temp file (can be ignored)
├── templates/              # HTML templates for the web interface
└── README.md
```

---

## Tech Stack

| Layer      | Technology             |
|------------|------------------------|
| Backend    | Python                 |
| Frontend   | HTML / CSS             |
| Web Server | Flask (or http.server) |
| Storage    | SQLite / local DB      |
| NLP/ML     | NLTK / scikit-learn    |

---

## Installation

### Prerequisites

- Python 3.8+
- pip

### Steps

1. **Clone the repository**

   ```bash
   git clone https://github.com/Prriiiyankaaa/Resume-Job-Analyzer.git
   cd Resume-Job-Analyzer
   ```

2. **Install dependencies**

   ```bash
   pip install -r requirements.txt
   ```

   > If a `requirements.txt` is not present, install common dependencies manually:
   > ```bash
   > pip install flask nltk scikit-learn PyMuPDF
   > ```

3. **Run the application**

   ```bash
   python resume_analyzer.py
   ```

4. **Open in your browser**

   ```
   http://localhost:5000
   ```

---

## Usage

1. Navigate to the app in your browser.
2. Upload your resume (PDF format).
3. Paste or enter the job description into the text field.
4. Click **Analyze** to generate your match score and keyword breakdown.
5. View past results by running:

   ```bash
   python view_db.py
   ```

---

## How It Works

1. **Resume Parsing** — Extracts raw text from the uploaded PDF resume.
2. **Keyword Extraction** — Identifies relevant skills and terms from both the resume and job description using NLP techniques.
3. **Similarity Scoring** — Computes a cosine similarity score using TF-IDF vectorization to quantify how well the resume matches the job posting.
4. **Storage** — Saves results (score, keywords, timestamp) to a local database for later review.

---
