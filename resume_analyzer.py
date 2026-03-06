import os
import tempfile

import google.genai as genai
import pdfplumber
from flask import Flask, jsonify, render_template, request

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "AIzaSyBHcQh87wsa1yRHPOnNCFW9l56QPxN7KOQ")
ALLOWED_EXTENSIONS = {"pdf"}

app = Flask(__name__)

client = genai.Client(api_key=GEMINI_API_KEY)


def read_resume(pdf_path):
    text = ""
    with pdfplumber.open(pdf_path) as pdf:
        for page in pdf.pages:
            page_text = page.extract_text()
            if page_text:
                text += page_text
    return text.strip()


def allowed_file(filename):
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS


def extract_uploaded_pdf_text(uploaded_file):
    suffix = ".pdf"
    temp_path = None
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            uploaded_file.save(tmp)
            temp_path = tmp.name
        return read_resume(temp_path)
    finally:
        if temp_path and os.path.exists(temp_path):
            os.remove(temp_path)


def analyze_resume_against_jd(resume_text, job_description):

    prompt = f"""
Persona: Senior Technical Recruiter.
Task: Analyze the retrieved resume against the JD.

Resume:
{resume_text}

Job Description:
{job_description}

Provide: Match Score, Gap Analysis, Google XYZ Bullet Points, and Interview Prep.
"""

    response = client.models.generate_content(
        model="models/gemini-2.5-flash",
        contents=prompt,
    )
    return response.text


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/analyze", methods=["POST"])
def analyze():
    job_description = (request.form.get("job_description") or "").strip()
    resume_file = request.files.get("resume")

    if resume_file is None or not resume_file.filename:
        return jsonify({"error": "Please upload a resume PDF."}), 400

    if not allowed_file(resume_file.filename):
        return jsonify({"error": "Only PDF files are allowed."}), 400

    if not job_description:
        return jsonify({"error": "job_description is required"}), 400

    try:
        resume_text = extract_uploaded_pdf_text(resume_file)
        if not resume_text:
            return jsonify({"error": "Could not extract text from the uploaded PDF."}), 400

        analysis = analyze_resume_against_jd(resume_text, job_description)
        return jsonify({"response": analysis})
    except Exception as exc:
        return jsonify({"error": str(exc)}), 500


if __name__ == "__main__":
    app.run(debug=True)
