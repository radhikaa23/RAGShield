import os
from pathlib import Path
import requests
import streamlit as st
from dotenv import load_dotenv

load_dotenv(Path(__file__).resolve().parents[1] / ".env")

st.set_page_config(page_title="RAGShield", page_icon="🛡️", layout="wide")
st.title("🛡️ RAGShield")
st.caption("Secure document intelligence · PDF answers with citations and retrieval safeguards")
st.info("The sample Employee Handbook is a template. Values in square brackets are placeholders, not confirmed company policy.")
api = os.getenv("RAGSHIELD_API_URL", "http://127.0.0.1:8000")
token = st.sidebar.text_input("Local access token", type="password", value=os.getenv("RAGSHIELD_ACCESS_TOKEN", ""))
headers = {"Authorization": f"Bearer {token}"} if token else {}
if not token:
    st.info("Set RAGSHIELD_ACCESS_TOKEN in .env, restart the API, then enter it here.")

def request(method, endpoint, **kwargs):
    try:
        response = requests.request(method, f"{api}{endpoint}", headers=headers, timeout=300, **kwargs)
        response.raise_for_status()
        return response.json()
    except requests.RequestException as exc:
        st.error(f"API request failed: {exc}")
        return None

st.subheader("Your documents")
max_pdf_mb = int(os.getenv("RAGSHIELD_MAX_PDF_MB", "20"))
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
docs = request("GET", "/documents") if token else []
docs = docs or []
selected = st.multiselect("Search only selected documents", docs, default=docs,
                          format_func=lambda item: item["filename"], key="selected_docs")

st.subheader("Ask the handbook")
question = st.text_input("Question", placeholder="How much sick leave is offered?")
control1, control2 = st.columns(2)
configured_k = int(os.getenv("RAGSHIELD_TOP_K", "5"))
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
