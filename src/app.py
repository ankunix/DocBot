import os
import pandas as pd
import streamlit as st

from sec_edgar.config import DATA_DIR
from langchain_core.documents import Document
from langchain_core.prompts import PromptTemplate
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface.embeddings import HuggingFaceEmbeddings
from langchain_huggingface.llms import HuggingFaceEndpoint
from langchain_community.vectorstores import Chroma
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain

# Set your Hugging Face API token (set as environment variable)
HF_TOKEN = os.environ.get("HF_TOKEN")

# Streamlit app title
st.title("📊 Financial Filings RAG Assistant")
st.write("Ask questions about company filings to gain financial insights.")

# Load and cache filings data
@st.cache_data
def load_data():
    data_path = os.path.join(DATA_DIR, "processed_filings.csv")
    df = pd.read_csv(data_path)
    return df

# Create Chroma vector store
@st.cache_resource
def create_vector_store(df):
    docs = []
    for _, row in df.iterrows():
        content = f"{row['Text']}\nTicker: {row['ticker']}\nCIK: {row['cik']}\nAccession Number: {row['accessionNumber']}"
        metadata = {
            "ticker": row["ticker"],
            "cik": row["cik"],
            "accessionNumber": row["accessionNumber"],
            "filepath": row["filepath"]
        }
        docs.append(Document(page_content=content, metadata=metadata))

    # Split into chunks
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    split_docs = splitter.split_documents(docs)

    # Use HuggingFace embeddings
    embedder = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True}
    )

    # Create Chroma vectorstore
    vectorstore = Chroma.from_documents(documents=split_docs, embedding=embedder)
    return vectorstore

# Build RAG chain
@st.cache_resource
def create_rag_chain(_vectorstore):
    retriever = _vectorstore.as_retriever(search_kwargs={"k": 5})

    prompt = PromptTemplate(
        template="""
You are a helpful assistant for analyzing SEC company filings.

Use ONLY the provided context to answer the question below. Always cite the ticker and accession number.

Context:
{context}

Question:
{input}

Answer:
""",
        input_variables=["context", "input"]
    )

    llm = HuggingFaceEndpoint(
        repo_id="mistralai/Mixtral-8x7B-Instruct-v0.1",
        token=HF_TOKEN,
        max_length=2048,
        temperature=0.1
    )

    doc_chain = create_stuff_documents_chain(llm, prompt)
    rag_chain = create_retrieval_chain(retriever, doc_chain)
    return rag_chain

# Main app logic
def main():
    df = load_data()
    st.success(f"✅ Loaded {len(df)} filings")

    vectorstore = create_vector_store(df)
    rag_chain = create_rag_chain(vectorstore)

    query = st.text_input("🔍 Ask about the filings:")
    if query:
        with st.spinner("Thinking..."):
            result = rag_chain.invoke({"input": query})
            st.subheader("📈 Answer")
            st.write(result["answer"])

if __name__ == "__main__":
    main()
