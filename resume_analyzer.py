import os
import tempfile
import uuid

import chromadb
import google.genai as genai
import pdfplumber
from flask import Flask, jsonify, render_template, request

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "YOUR API KEY")
ALLOWED_EXTENSIONS = {"pdf"}
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CHROMA_PATH = os.path.join(BASE_DIR, "resume_vault")

app = Flask(__name__)

client = genai.Client(api_key=GEMINI_API_KEY)
chroma_client = chromadb.PersistentClient(path=CHROMA_PATH)


class GeminiEmbeddingFunction:
    def __call__(self, input):
        embeddings = []
        for text in input:
            result = client.models.embed_content(
                model="models/gemini-embedding-001",
                contents=text,
            )
            embeddings.append(result.embeddings[0].values)
        return embeddings

    def embed_query(self, input):
        if isinstance(input, str):
            input = [input]
        return self.__call__(input)

    def embed_documents(self, input):
        return self.__call__(input)

    def name(self):
        return "gemini-embedding-001"


collection = chroma_client.get_or_create_collection(
    name="resume_chunks",
    embedding_function=GeminiEmbeddingFunction(),
)


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


def chunk_text(text, chunk_size=1200, overlap=200):
    chunks = []
    start = 0
    text_length = len(text)

    while start < text_length:
        end = min(start + chunk_size, text_length)
        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)
        if end == text_length:
            break
        start = max(0, end - overlap)

    return chunks


def store_resume_chunks(resume_text, original_filename):
    chunks = chunk_text(resume_text)
    doc_set_id = str(uuid.uuid4())

    ids = [f"{doc_set_id}_{index}" for index in range(len(chunks))]
    metadatas = [
        {
            "doc_set_id": doc_set_id,
            "filename": original_filename,
            "chunk_index": index,
        }
        for index in range(len(chunks))
    ]

    collection.upsert(
        ids=ids,
        documents=chunks,
        metadatas=metadatas,
    )
    return doc_set_id, len(chunks)


def retrieve_relevant_chunks(job_description, doc_set_id, n_results=4):
    results = collection.query(
        query_texts=[job_description],
        where={"doc_set_id": doc_set_id},
        n_results=n_results,
        include=["documents", "distances"],
    )

    documents = results.get("documents", [[]])[0]
    distances = results.get("distances", [[]])[0]
    context = "\n\n".join(documents)
    return context, distances, len(documents)


def analyze_resume_against_jd(resume_context, job_description):

    prompt = f"""
    Rules:
    -Return ONLY valid response, no markdown, no backticks, no extra text, no astrick, no bold, no italics, no underline, no *.
Persona: Senior Technical Recruiter.
Task: Analyze the retrieved resume context (retrieved from vector database) against the JD.

Resume:
{resume_context}

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

        doc_set_id, chunk_count = store_resume_chunks(resume_text, resume_file.filename)
        retrieved_context, distances, retrieved_chunks = retrieve_relevant_chunks(
            job_description,
            doc_set_id,
            n_results=min(4, chunk_count),
        )

        if not retrieved_context:
            return jsonify({"error": "No relevant resume chunks found in vector database."}), 400

        analysis = analyze_resume_against_jd(retrieved_context, job_description)

        avg_distance = None
        if distances:
            avg_distance = round(float(sum(distances) / len(distances)), 4)

        return jsonify(
            {
                "response": analysis,
                "vector_db": "ChromaDB",
                "chunks_indexed": chunk_count,
                "chunks_retrieved": retrieved_chunks,
                "avg_distance": avg_distance,
            }
        )
    except Exception as exc:
        return jsonify({"error": str(exc)}), 500


if __name__ == "__main__":
    app.run(debug=True)
