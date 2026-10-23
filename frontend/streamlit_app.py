import os
from pathlib import Path
import requests
import streamlit as st
from dotenv import load_dotenv

load_dotenv(Path(__file__).resolve().parents[1] / ".env")


def setting(name: str, default: str = "") -> str:
    """Read Streamlit Cloud secrets first, then local environment/.env values."""
    try:
        value = st.secrets.get(name)
        if value is not None:
            return str(value)
    except Exception:
        # Local runs may not have a .streamlit/secrets.toml file.
        pass
    return os.getenv(name, default)


def enabled(name: str, default: str = "false") -> bool:
    return setting(name, default).strip().lower() in {"1", "true", "yes", "on"}

st.set_page_config(page_title="RAGShield", page_icon="🛡️", layout="wide")
st.title("🛡️ RAGShield")
st.caption("Secure document intelligence · PDF answers with citations and retrieval safeguards")
st.info("The sample Employee Handbook is a template. Values in square brackets are placeholders, not confirmed company policy.")
api = setting("RAGSHIELD_API_URL", "http://127.0.0.1:8000").rstrip("/")
demo_mode = enabled("RAGSHIELD_DEMO_MODE")
token = setting("RAGSHIELD_ACCESS_TOKEN")
headers = {"Authorization": f"Bearer {token}"} if token else {}
if not token:
    st.error("Access token is not configured. Set it in .env locally or in the app's Streamlit Cloud secrets.")
if demo_mode:
    st.info("Read-only portfolio demo: only pre-indexed sample documents can be queried; uploads are disabled.")
elif token:
    token = st.sidebar.text_input("Local access token", type="password", value=token)
    headers = {"Authorization": f"Bearer {token}"} if token else {}

def request(method, endpoint, **kwargs):
    try:
        response = requests.request(method, f"{api}{endpoint}", headers=headers, timeout=300, **kwargs)
        response.raise_for_status()
        return response.json()
    except requests.RequestException as exc:
        st.error(f"API request failed: {exc}")
        return None

st.subheader("Your documents")
if not demo_mode:
    max_pdf_mb = int(setting("RAGSHIELD_MAX_PDF_MB", "20"))
    uploads = st.file_uploader(f"Upload one or more PDFs (maximum {max_pdf_mb} MB each)", type=["pdf"],
                               accept_multiple_files=True, max_upload_size=max_pdf_mb)
    if st.button("Index PDFs", disabled=not uploads or not token):
        for upload in uploads:
            result = request("POST", "/documents", files={"file": (upload.name, upload.getvalue(), "application/pdf")})
            if result:
                if result.get("already_indexed"):
                    st.info(f"{result['filename']} is already indexed. No duplicate chunks were added.")
                else:
                    st.success(f"Indexed {result['filename']} · {result['pages']} pages · {result['flagged_chunks']} injection-flagged chunks excluded · {result['redacted_secret_chunks']} chunks with redacted secrets")
else:
    st.caption("Document upload is disabled in public demo mode.")
docs = request("GET", "/documents") if token else []
docs = docs or []
selected = st.multiselect("Search only selected documents", docs, default=docs,
                          format_func=lambda item: item["filename"], key="selected_docs")

st.subheader("Ask the handbook")
question = st.text_input("Question", placeholder="How much sick leave is offered?")
control1, control2 = st.columns(2)
configured_k = int(setting("RAGSHIELD_TOP_K", "5"))
k = control1.select_slider("Retrieved passages (K)", options=[3, 5, 10],
                           value=configured_k if configured_k in [3, 5, 10] else 5)
strategy = control2.selectbox("Retrieval method", ["semantic", "keyword", "hybrid"],
    help="Semantic finds related wording; keyword helps with exact names, IDs, and error text; hybrid merges both with Reciprocal Rank Fusion.")
if st.button("Ask", disabled=not token or not question or not selected):
    with st.spinner("Searching safe, relevant evidence..."):
        result = request("POST", "/query", json={"question": question, "doc_ids": [d["doc_id"] for d in selected], "k": k, "strategy": strategy})
    if result:
        st.markdown("### Answer")
        st.write(result["answer"])
        left, right = st.columns(2)
        left.metric("Retrieved passages", len(result["sources"]))
        right.metric("Generation", result["provider"])
        if result.get("tokens") is not None:
            st.caption(f"LLM tokens: {result['tokens']}")
        if result.get("warning"):
            st.warning(result["warning"])
        with st.expander("Citations used in the answer", expanded=True):
            if result["cited_sources"]:
                for src in result["cited_sources"]:
                    st.markdown(f"**{src['filename']} · page {src['page']}** · {src['source_id']}")
                    st.caption(src["text"])
            else:
                st.write("The answer did not cite a source passage.")
        with st.expander("All retrieved passages"):
            if result["sources"]:
                for src in result["sources"]:
                    score = (f"cosine distance {src['distance']} · similarity {src['cosine_similarity']}"
                             if src['distance'] is not None else f"BM25/RRF score {src['retrieval_score']}")
                    st.markdown(f"**{src['filename']} · page {src['page']}** · {score}")
                    st.caption(f"Document {src['doc_id']} · chunk {src['chunk_id']}")
                    st.caption(src["text"])
            else:
                st.write("No relevant safe sources found.")

with st.expander("Security controls"):
    st.markdown("""
    - Uploaded PDFs are treated as untrusted evidence, never as instructions.
    - Prompt-injection-like chunks are flagged and excluded from retrieval.
    - Retrieval is filtered by owner and the documents selected above, then by relevance.
    - API access requires a bearer token; suspected secrets are redacted before indexing and from answers.
    - The UI reports page citations and token usage where the configured model provides it.
    """)
