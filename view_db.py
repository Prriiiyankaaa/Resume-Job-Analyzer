import streamlit as st
import pdfplumber
import pandas as pd
import chromadb
import google.genai as genai
from chromadb.api.types import EmbeddingFunction, Documents, Embeddings

# --------------------------------------------------
# STREAMLIT CONFIG (ONLY ONCE)
# --------------------------------------------------
st.set_page_config(page_title="AI Resume Vault", layout="wide")
st.title("📁 Resume Vault Explorer")

# --------------------------------------------------
# CONFIG
# --------------------------------------------------
GEMINI_API_KEY = "AIzaSyA18zHEIeEUPPSNGUgvMqY8caydKOV9V8s"
CHROMA_PATH = "./resume_vault"

# --------------------------------------------------
# CLIENTS (IMPORTANT: DIFFERENT NAMES)
# --------------------------------------------------
chroma_client = chromadb.PersistentClient(path=CHROMA_PATH)
gemini_client = genai.Client(api_key=GEMINI_API_KEY)

# --------------------------------------------------
# GEMINI EMBEDDING FUNCTION
# --------------------------------------------------
class GeminiEmbeddingFunction(EmbeddingFunction):
    def __call__(self, input: Documents) -> Embeddings:
        embeddings = []
        for text in input:
            result = gemini_client.models.embed_content(
                model="text-embedding-004",
                contents=text
            )
            embeddings.append(result.embeddings[0].values)
        return embeddings

    def name(self) -> str:
        return "GeminiEmbeddingFunction"

# --------------------------------------------------
# GET / CREATE COLLECTION
# --------------------------------------------------
collection = chroma_client.get_or_create_collection(
    name="resumes",
    embedding_function=GeminiEmbeddingFunction()
)

# --------------------------------------------------
# SIDEBAR – COLLECTION VIEWER
# --------------------------------------------------
with st.sidebar:
    st.header("📚 Collections")
    collections = chroma_client.list_collections()
    col_names = [c.name for c in collections]

    selected_col = st.selectbox(
        "Select Collection",
        col_names if col_names else ["resumes"]
    )

    st.divider()
    st.header("⬆ Upload Center")
    uploaded_file = st.file_uploader("Upload Resume (PDF)", type="pdf")
    jd_input = st.text_area("Paste Job Description", height=200)
    analyze_btn = st.button("🚀 Analyze Match")

# --------------------------------------------------
# UPLOAD + ANALYZE LOGIC
# --------------------------------------------------
if uploaded_file and jd_input and analyze_btn:
    with st.spinner("Processing resume & running AI analysis..."):

        # ---- Extract PDF Text ----
        resume_text = ""
        with pdfplumber.open(uploaded_file) as pdf:
            for page in pdf.pages:
                resume_text += page.extract_text() or ""

        # ---- Store in ChromaDB ----
        doc_id = uploaded_file.name.replace(" ", "_")

        collection.upsert(
            documents=[resume_text],
            ids=[doc_id],
            metadatas=[{"filename": uploaded_file.name}]
        )

        # ---- Similarity Search ----
        search_results = collection.query(
            query_texts=[jd_input],
            n_results=3
        )

        # ---- Gemini Analysis ----
        prompt = f"""
Analyze the resume against the job description.

Resume:
{resume_text}

Job Description:
{jd_input}

Give strengths, weaknesses, and a match score (0–100).
"""

        response = gemini_client.models.generate_content(
            model="gemini-2.0-flash-exp",
            contents=prompt
        )

        # ---- DISPLAY ----
        st.success("✅ Analysis Complete")

        col1, col2 = st.columns(2)

        with col1:
            st.subheader("📄 Resume Text (Preview)")
            st.text(resume_text[:1500] + "...")

        with col2:
            st.subheader("🤖 AI Feedback")
            st.markdown(response.text)

# --------------------------------------------------
# COLLECTION VIEW + SEARCH
# --------------------------------------------------
if selected_col:
    st.divider()
    st.subheader(f"📂 Collection: {selected_col}")

    active_collection = chroma_client.get_collection(name=selected_col)
    data = active_collection.get(include=["documents", "metadatas"])

    st.write(f"**Total Records:** {len(data['ids'])}")

    if data["ids"]:
        df = pd.DataFrame({
            "ID": data["ids"],
            "Snippet": [doc[:200] + "..." for doc in data["documents"]],
            "Metadata": [str(m) for m in data["metadatas"]],
        })
        st.dataframe(df, use_container_width=True)

    # ---- Semantic Search ----
    st.subheader("🔍 Semantic Search")
    query = st.text_input("Search resumes by meaning")

    if query:
        results = active_collection.query(
            query_texts=[query],
            n_results=3
        )

        st.write("### Top Matches")
        for i, doc in enumerate(results["documents"][0]):
            st.markdown(f"**Result {i+1}:**")
            st.write(doc[:500] + "...")
