import os
import tempfile
import uuid
import hashlib
import re 

import chromadb
import google.genai as genai
import pdfplumber
from flask import Flask, jsonify, render_template, request

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "YOUR API KEY")
ALLOWED_EXTENSIONS = {"pdf"}
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CHROMA_PATH = os.path.join(BASE_DIR, "resume_vault")


JD_MIN_CHARS = 80
JD_MIN_WORDS = 15
JD_MIN_UNIQUE_WORDS = 10
JD_MIN_ENTROPY = 3.5          
RESUME_MIN_CHARS = 150
RESUME_MIN_WORDS = 40
RESUME_KEYWORD_THRESHOLD = 3  
RESUME_KEYWORDS = {
    "experience", "education", "skills", "work", "project", "projects",
    "university", "college", "degree", "engineer", "developer", "designer",
    "manager", "analyst", "intern", "internship", "company", "organization",
    "responsible", "developed", "managed", "led", "built", "implemented",
    "bachelor", "master", "certification", "employment", "professional",
    "objective", "summary", "profile", "achievements", "accomplishments",
    "technologies", "languages", "framework", "database", "software",
    "graduated", "gpa", "cgpa", "volunteer", "publications", "research",
}

JD_DOMAIN_KEYWORDS = {
    "engineer", "developer", "designer", "manager", "analyst", "scientist",
    "architect", "lead", "senior", "junior", "full", "stack", "frontend",
    "backend", "data", "machine", "learning", "software", "hardware",
    "python", "java", "javascript", "react", "node", "sql", "cloud",
    "aws", "azure", "gcp", "devops", "agile", "scrum", "product",
    "experience", "required", "preferred", "qualification", "skills",
    "responsibility", "responsibilities", "role", "team", "work",
    "position", "job", "candidate", "hire", "hiring", "apply", "salary",
    "degree", "bachelor", "master", "phd", "communication", "collaborate",
    "years", "minimum", "bonus", "remote", "hybrid", "onsite",
}

app = Flask(__name__)
client = genai.Client(api_key=GEMINI_API_KEY)
chroma_client = chromadb.PersistentClient(path=CHROMA_PATH)



def shannon_entropy(text: str) -> float:
    """Character-level Shannon entropy (bits). Low → repetitive / mashed text."""
    text = text.lower()
    freq = {}
    for ch in text:
        freq[ch] = freq.get(ch, 0) + 1
    total = len(text)
    return -sum((c / total) * log2(c / total) for c in freq.values() if c)


def validate_job_description(jd: str) -> tuple[bool, str | None]:
    """
    Return (True, None) if the JD looks legitimate,
    or (False, <human-readable error message>) otherwise.
    """
    words = jd.split()
    word_count = len(words)
    unique_words = {w.lower().strip(".,;:!?\"'") for w in words}

    if len(jd) < JD_MIN_CHARS:
        return False, (
            f"Job description is too short ({len(jd)} characters). "
            "Please paste a real job description with at least a few sentences."
        )

    if word_count < JD_MIN_WORDS:
        return False, (
            f"Job description only has {word_count} word(s). "
            "A valid job description should have at least 15 words."
        )

    if len(unique_words) < JD_MIN_UNIQUE_WORDS:
        return False, (
            "Job description has too little word variety — it may be repeated text or placeholder content. "
            "Please enter a real job description."
        )

    entropy = shannon_entropy(jd)
    if entropy < JD_MIN_ENTROPY:
        return False, (
            "Job description appears to contain random or repetitive characters. "
            "Please paste a real job description."
        )

    # At least a few domain-relevant words must appear
    jd_lower = jd.lower()
    domain_hits = sum(1 for kw in JD_DOMAIN_KEYWORDS if kw in jd_lower)
    if domain_hits < 2:
        return False, (
            "Job description doesn't appear to describe a real job role. "
            "Please paste a full job posting (title, responsibilities, requirements, etc.)."
        )

    return True, None


def validate_resume_text(text: str) -> tuple[bool, str | None]:
    """
    Return (True, None) if the extracted text looks like a resume,
    or (False, <human-readable error message>) otherwise.
    """
    if len(text) < RESUME_MIN_CHARS:
        return False, (
            "The uploaded PDF appears to be empty or contains almost no text. "
            "Please upload a text-based resume PDF (not a scanned image)."
        )

    words = text.split()
    if len(words) < RESUME_MIN_WORDS:
        return False, (
            f"Only {len(words)} words were extracted from the PDF. "
            "The file may be image-only. Please upload a text-selectable resume PDF."
        )

    text_lower = text.lower()
    keyword_hits = sum(1 for kw in RESUME_KEYWORDS if kw in text_lower)
    if keyword_hits < RESUME_KEYWORD_THRESHOLD:
        return False, (
            "The uploaded PDF does not appear to be a resume. "
            "Please upload a CV or resume containing sections like Experience, Education, or Skills."
        )

    return True, None


# ── Embedding ──────────────────────────────────────────────────────────────────

class GeminiEmbeddingFunction:
    """
    Wraps the Gemini embedding API.

    Key optimisation: embed ALL chunks in a SINGLE batched API call instead of
    one call per chunk (the original code made N round-trips for N chunks).
    """

    MODEL = "models/gemini-embedding-001"

    def __call__(self, input: list[str]) -> list[list[float]]:
        if not input:
            return []
        # Single batched request — dramatically faster than N serial requests
        result = client.models.embed_content(
            model=self.MODEL,
            contents=input,
        )
        return [emb.values for emb in result.embeddings]

    # ChromaDB calls these two methods
    def embed_query(self, input):
        return self.__call__([input] if isinstance(input, str) else input)

    def embed_documents(self, input):
        return self.__call__(input)

    def name(self):
        return "gemini-embedding-001"


embedding_fn = GeminiEmbeddingFunction()

collection = chroma_client.get_or_create_collection(
    name="resume_chunks",
    embedding_function=embedding_fn,
)


# ── PDF helpers ────────────────────────────────────────────────────────────────

def allowed_file(filename: str) -> bool:
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS


def extract_pdf_text(uploaded_file) -> str:
    """Save upload to a temp file, extract text, clean up."""
    temp_path = None
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
            uploaded_file.save(tmp)
            temp_path = tmp.name

        text = ""
        with pdfplumber.open(temp_path) as pdf:
            for page in pdf.pages:
                page_text = page.extract_text()
                if page_text:
                    text += page_text
        return text.strip()
    finally:
        if temp_path and os.path.exists(temp_path):
            os.remove(temp_path)


# ── Chunking + ChromaDB ────────────────────────────────────────────────────────

def chunk_text(text: str, chunk_size: int = 600) -> list[str]: 
    sentences = re.split(r'(?<+[.!?) +', text)
    
    chunks = []
    current_chunk = ""

     for sentence in sentences:
        if len(current_chunk) + len(sentence) < chunk_size:
            current_chunk += " " + sentence
        else:
            chunks.append(current_chunk.strip())
            current_chunk = sentence

    if current_chunk:
        chunks.append(current_chunk.strip())
        
    # length = len(text)
    # while start < length:
    #     end = min(start + chunk_size, length)
    #     chunk = text[start:end].strip()
    #     if chunk:
    #         chunks.append(chunk)
    #     if end == length:
    #         break
    #     start = max(0, end - overlap)

    return chunks


def store_resume_chunks(resume_text: str, original_filename: str) -> tuple[str, int]:
    """
    Upsert resume chunks into ChromaDB.
    Returns (doc_set_id, chunk_count).

    Uses content hash so the same resume is never re-embedded — the expensive
    batch embed call is skipped entirely on cache hits.
    """
    doc_set_id = hashlib.md5(resume_text.encode()).hexdigest()

    existing = collection.get(where={"doc_set_id": doc_set_id})
    if existing["ids"]:
        return doc_set_id, len(existing["ids"])

    chunks = chunk_text(resume_text)
    ids = [f"{doc_set_id}_{i}" for i in range(len(chunks))]
    metadatas = [
        {"doc_set_id": doc_set_id, 
         "filename": original_filename, 
         "chunk_index": i,
         "chunk_length": len(chunks[i])
        }
        for i in range(len(chunks))
    ]

    # One batch embed call covers ALL chunks
    collection.upsert(ids=ids, documents=chunks, metadatas=metadatas)
    return doc_set_id, len(chunks)


# def retrieve_relevant_chunks(
#     job_description: str, doc_set_id: str, n_results: int = 6
# ) -> tuple[str, list[float], int]:
#     results = collection.query(
#         query_texts=[job_description],
#         where={"doc_set_id": doc_set_id},
#         n_results=n_results,
#         include=["documents", "distances"],
#     )
def retrieve_relevant_chunks(job_description, doc_set_id, n_results=6):
    results = collection.query(
        query_texts=[job_description],
        where={"doc_set_id": doc_set_id},
        n_results=n_results,
        include=["documents", "distances"],
    )
    
    documents = results.get("documents", [[]])[0]
    distances = results.get("distances", [[]])[0]
 # sort by relevance
    sorted_pairs = sorted(zip(documents, distances), key=lambda x: x[1])

    top_docs = [doc for doc, _ in sorted_pairs[:4]]

    # return "\n\n".join(documents), distances, len(documents)
 return "\n\n".join(top_docs), distances, len(top_docs)



# ── LLM analysis ──────────────────────────────────────────────────────────────

def analyze_resume_against_jd(resume_context: str, job_description: str) -> str:
    prompt = f"""You are a strict ATS (Applicant Tracking System) acting as a Senior Technical Recruiter.

RULES:
- Score out of 100. Be extremely critical.
- Base your entire answer ONLY on the provided resume context. Do NOT hallucinate or assume skills.
- If skills are missing, state them explicitly.
- Do NOT use markdown, asterisks, bold, italics, bullet symbols, or backticks.
- Write in plain text only.

TASK:
Analyze how well the resume matches the job description below.

Resume Context:
{resume_context}

Job Description:
{job_description}

OUTPUT FORMAT (plain text, no markdown):
Match Score Out of 100: [0-100]

Gap Analysis:
Missing skills: [list]
Weak areas: [list]

Relevant Matches:
Matching skills: [list]

Final Verdict:
[Strong / Moderate / Weak fit — one sentence explanation]

Interview Prep (top 3 questions likely to be asked based on gaps):
1. [question]
2. [question]
3. [question]
"""
    response = client.models.generate_content(
        model="gemini-3.1-flash-lite-preview",
        contents=prompt,
    )
    return response.text


# ── Routes ─────────────────────────────────────────────────────────────────────

@app.route("/")
def index():
    return render_template("index.html")


@app.route("/analyze", methods=["POST"])
def analyze():
    job_description = (request.form.get("job_description") or "").strip()
    resume_file = request.files.get("resume")

    # ── Basic presence checks ─────────────────────────────────────────────────
    if resume_file is None or not resume_file.filename:
        return jsonify({"error": "Please upload a resume PDF."}), 400

    if not allowed_file(resume_file.filename):
        return jsonify({"error": "Only PDF files are allowed."}), 400

    if not job_description:
        return jsonify({"error": "Please enter a job description."}), 400

    # ── Job description validation (fast, no API calls) ───────────────────────
    jd_ok, jd_error = validate_job_description(job_description)
    if not jd_ok:
        return jsonify({"error": jd_error}), 422

    try:
        # ── Extract text ──────────────────────────────────────────────────────
        resume_text = extract_pdf_text(resume_file)

        # ── Resume text validation (fast, no API calls) ───────────────────────
        resume_ok, resume_error = validate_resume_text(resume_text)
        if not resume_ok:
            return jsonify({"error": resume_error}), 422

        # ── Embed + store (batched) ────────────────────────────────────────────
        doc_set_id, chunk_count = store_resume_chunks(resume_text, resume_file.filename)

        # ── Retrieve relevant chunks ──────────────────────────────────────────
        retrieved_context, distances, retrieved_chunks = retrieve_relevant_chunks(
            job_description,
            doc_set_id,
            n_results=min(6, chunk_count),
        )

        if not retrieved_context:
            return jsonify({"error": "No relevant resume content found. Please try again."}), 400

        # ── LLM analysis ──────────────────────────────────────────────────────
        analysis = analyze_resume_against_jd(retrieved_context, job_description)

        avg_distance = (
            round(sum(distances) / len(distances), 4) if distances else None
        )

        return jsonify({
            "response": analysis,
            "vector_db": "ChromaDB",
            "chunks_indexed": chunk_count,
            "chunks_retrieved": retrieved_chunks,
            "avg_distance": avg_distance,
        })

    except Exception as exc:
        return jsonify({"error": str(exc)}), 500


if __name__ == "__main__":
    app.run(debug=True)
