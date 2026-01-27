import PyPDF2
import chromadb
from chromadb.utils import embedding_functions
from sentence_transformers import SentenceTransformer, util

# 1. Setup Hugging Face Embedding Function for ChromaDB
# This model is lightweight and excellent for semantic similarity
model_name = "all-MiniLM-L6-v2"
hf_ef = embedding_functions.SentenceTransformerEmbeddingFunction(model_name=model_name)

# 2. Initialize ChromaDB (Ephemeral/In-Memory for this example)
client = chromadb.Client()
collection = client.get_or_create_collection(name="resume_analysis", embedding_function=hf_ef)

def extract_text_from_pdf(pdf_path):
    """Extracts all text from a PDF file."""
    text = ""
    with open(pdf_path, "rb") as f:
        reader = PyPDF2.PdfReader(f)
        for page in reader.pages:
            content = page.extract_text()
            if content:
                text += content
    return text
count = 0
def analyze_resume(resume_pdf_path, job_description_text):
    global count
    count += 1
    # Extract text from resume
    resume_text = extract_text_from_pdf(resume_pdf_path)
    
    resume_id = "resume_unique_01"
    
    existing = collection.get(ids=[resume_id])
    
    if not existing['ids']:
        print("New resume detected. Extracting and embedding...")
        resume_text = extract_text_from_pdf(resume_pdf_path)
        collection.add(documents=[resume_text], ids=[resume_id])
    else:
        print("Resume found in local vector storage. Skipping extraction.")

    # Query logic remains the same
    results = collection.query(
        query_texts=[job_description_text],
        n_results=1,
        include=["distances"]
    )
    
    # Calculate and display score
    distance = results['distances'][0][0]
    similarity_score = max(0, (1 - distance) * 100)
    print(f"Match Accuracy: {similarity_score:.2f}%")
    
    if similarity_score > 70:
        print("Status: Strong Match")
    elif similarity_score > 40:
        print("Status: Partial Match")
    else:
        print("Status: Low Correlation")

# --- Example Usage ---
if __name__ == "__main__":
    # Path to your resume PDF
    resume_path = "resume.pdf" 
    
    # Job Description as a paragraph
    jd = """
    We are looking for a Software Engineer with experience in Python, 
    React, and Vector Databases. Candidates should have a strong 
    understanding of Machine Learning APIs and cloud deployments.
    """
    
    try:
        analyze_resume(resume_path, jd)
    except FileNotFoundError:
        print("Error: Please provide a valid path to a 'resume.pdf' file.")