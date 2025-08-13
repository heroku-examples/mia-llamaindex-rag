import streamlit as st
import time
import os
from PIL import Image
from pathlib import Path
from llama_index.llms.heroku import Heroku
from llama_index.embeddings.openai_like import OpenAILikeEmbedding
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader, Settings, StorageContext
from llama_index.core.node_parser import SentenceSplitter
from llama_index.vector_stores.postgres import PGVectorStore
from dotenv import load_dotenv
from sqlalchemy import make_url
from utils import show_code

load_dotenv()

# Get Heroku AI and Postgres config from environment variables
inference_model_id = os.environ.get("INFERENCE_MODEL_ID")
embedding_model_id = os.environ.get("EMBEDDING_MODEL_ID")

# Initialize session state
if 'messages' not in st.session_state:
    st.session_state.messages = []
if 'documents_uploaded' not in st.session_state:
    st.session_state.documents_uploaded = []
if 'model_id' not in st.session_state:
    st.session_state.model_id = inference_model_id
if 'embedding_model' not in st.session_state:
    st.session_state.embedding_model = embedding_model_id
if 'top_k_results' not in st.session_state:
    st.session_state.top_k_results = 10
if 'response_mode' not in st.session_state:
    st.session_state.response_mode = "tree_summarize"
if 'query_engine' not in st.session_state:
    st.session_state.query_engine = None
if 'index_ready' not in st.session_state:
    st.session_state.index_ready = False

# Get current RAG settings from session state
top_k = st.session_state.top_k_results
response_mode = st.session_state.response_mode

# LlamaIndex RAG Pipeline


def setup_rag_index():
    """Setup the RAG index and query engine - run once when documents are available"""

    # Initialize Heroku AI model
    llm = Heroku()

    # Use OpenAILikeEmbedding class pointed at Heroku's service
    # It reads the EMBEDDING_URL, EMBEDDING_KEY, and EMBEDDING_MODEL_ID from env vars
    embed_model = OpenAILikeEmbedding(
        # api_base="http://localhost:3000/v1",
        api_base=os.environ.get("EMBEDDING_URL") + "/v1",
        api_key=os.environ.get("EMBEDDING_KEY"),
        model_name=os.environ.get("EMBEDDING_MODEL_ID"),
        embed_batch_size=96,
    )

    # Configure text splitter to ensure chunks are under the character limit
    # Using 512 to leave room for metadata and ensure we stay under limit
    text_splitter = SentenceSplitter(
        chunk_size=512,  # Max characters per chunk
        chunk_overlap=10,  # Overlap between chunks for context
        separator=" ",
    )

    # Set global settings for LlamaIndex
    Settings.text_splitter = text_splitter
    Settings.embed_model = embed_model
    Settings.llm = llm

    # Load data from the documents folder
    documents = SimpleDirectoryReader(DOCUMENTS_FOLDER).load_data()

    # Parse documents into nodes with proper chunking
    nodes = text_splitter.get_nodes_from_documents(documents)

    # Connect to the Heroku Postgres database with pgvector support
    # The DATABASE_URL config var is automatically set by Heroku and contains the connection string
    # We need to parse the connection string and pass it to the PGVectorStore
    database_url = os.environ.get("DATABASE_URL").replace(
        "postgres://", "postgresql://")
    url = make_url(database_url)
    vector_store = PGVectorStore.from_params(
        database=url.database,
        host=url.host,
        port=url.port,
        user=url.username,
        password=url.password,
        table_name="documents",
        embed_dim=1024  # Cohere embeddings have a dimension of 1024
    )

    # Create a storage context to store the index
    storage_context = StorageContext.from_defaults(vector_store=vector_store)

    # Create the index from nodes (already chunked properly)
    index = VectorStoreIndex(
        nodes=nodes,
        vector_store=vector_store,
        embed_model=embed_model,
        storage_context=storage_context,
        show_progress=True
    )

    # Create a query engine to interact with the data
    query_engine = index.as_query_engine(
        llm=llm,
        response_mode=response_mode, # "tree_summarize", "refine", "compact", "simple_summarize"
        similarity_top_k=top_k  # Number of relevant document chunks to retrieve
    )

    # Use the query engine to query the documents
    # Example:
    # response = query_engine.query("What is the main takeaway from my documents?")
    return query_engine


def query_documents(prompt, query_engine):
    """Query the documents using the pre-built query engine"""
    response = query_engine.query(prompt)
    return response


# Load Heroku AI icon
icon = Image.open("assets/mia-icon.png")

# Page config
st.set_page_config(
    page_title="RAG Chat with LlamaIndex and Heroku AI",
    page_icon=icon,
    layout="wide"
)
st.logo(icon, size="large", link="https://www.heroku.com/ai")

# Hide Deploy Button and other elements
st.markdown("""
    <style>
        .reportview-container {
            margin-top: -2em;
        }
        #MainMenu {visibility: hidden;}
        .stAppDeployButton {display:none;}
        footer {visibility: hidden;}
        #stDecoration {display:none;}
    </style>
""", unsafe_allow_html=True)

# Constants
DOCUMENTS_FOLDER = Path("documents")
DOCUMENTS_FOLDER.mkdir(exist_ok=True)  # Ensure folder exists

# Helper functions


def get_documents_in_folder():
    """Get list of documents in the documents folder"""
    if not DOCUMENTS_FOLDER.exists():
        return []
    return [f.name for f in DOCUMENTS_FOLDER.iterdir() if f.is_file() and f.suffix.lower() in ['.txt', '.pdf', '.docx', '.md', '.csv']]


def save_uploaded_file(uploaded_file):
    """Save uploaded file to documents folder"""
    file_path = DOCUMENTS_FOLDER / uploaded_file.name
    with open(file_path, "wb") as f:
        f.write(uploaded_file.getbuffer())
    return file_path


# Sidebar
with st.sidebar:
    st.title("RAG Chat Settings")

    # Model Configuration Section
    st.subheader("🤖 Model Configuration")

    model_options = [
        inference_model_id,
    ]
    st.session_state.model_id = st.selectbox(
        "Model ID",
        model_options,
        index=model_options.index(st.session_state.model_id)
    )

    embedding_options = [
        embedding_model_id,
    ]
    st.session_state.embedding_model = st.selectbox(
        "Embedding Model",
        embedding_options,
        index=embedding_options.index(st.session_state.embedding_model)
    )

    st.divider()

    # Retrieval Settings Section
    st.subheader("🔍 Retrieval Settings")

    # Store previous values to detect changes
    prev_top_k = st.session_state.get(
        'prev_top_k', st.session_state.top_k_results)
    prev_response_mode = st.session_state.get(
        'prev_response_mode', st.session_state.response_mode)

    st.session_state.top_k_results = st.slider(
        "Top K Results",
        min_value=1,
        max_value=20,
        value=st.session_state.top_k_results,
        help="Number of relevant document chunks to retrieve"
    )

    response_modes = [
        "tree_summarize",
        "refine",
        "compact",
        "simple_summarize"
    ]
    st.session_state.response_mode = st.selectbox(
        "Response Mode",
        response_modes,
        index=response_modes.index(st.session_state.response_mode),
        help="How to combine retrieved chunks into final response"
    )

    # Check if settings changed and mark index for rebuild
    if (st.session_state.top_k_results != prev_top_k or
            st.session_state.response_mode != prev_response_mode):
        if st.session_state.index_ready:
            st.session_state.index_ready = False
            st.session_state.query_engine = None
            st.info("⚙️ Settings changed - index will be rebuilt")

    # Update previous values
    st.session_state.prev_top_k = st.session_state.top_k_results
    st.session_state.prev_response_mode = st.session_state.response_mode

    st.divider()

    # Reset Chat Button
    if st.button("🗑️ Reset Chat", use_container_width=True):
        st.session_state.messages = []
        st.rerun()

# Main content area
st.title("🤖 RAG Chat with LlamaIndex and Heroku AI")

# Documents Overview Section
upload_container, documents_container = st.columns(2)

with upload_container:
    st.subheader("📁 Upload Documents")
    uploaded_files = st.file_uploader(
        "Choose files to upload for RAG",
        accept_multiple_files=True,
        type=['txt', 'pdf', 'docx', 'md', 'csv']
    )

with documents_container:
    st.subheader("📋 Documents Folder")
    available_docs = get_documents_in_folder()
    if available_docs:
        st.info(f"Found {len(available_docs)} document(s) in folder")
        for doc in available_docs:
            st.text(f" 📄 {doc}")

    else:
        st.warning("No documents in folder")


if uploaded_files:
    for uploaded_file in uploaded_files:
        if uploaded_file.name not in st.session_state.documents_uploaded:
            with st.spinner(f"Processing {uploaded_file.name}..."):
                # Save file to documents folder
                file_path = save_uploaded_file(uploaded_file)
                time.sleep(1)  # Simulate processing

                st.session_state.documents_uploaded.append(uploaded_file.name)
                st.success(
                    f"✅ Successfully saved: {uploaded_file.name} to documents folder")

                # Reset index status to trigger rebuild
                st.session_state.index_ready = False
                st.session_state.query_engine = None
                st.rerun()  # Refresh to show updated file tree

# Setup RAG index if we have documents and index isn't ready
available_docs = get_documents_in_folder()
if available_docs and not st.session_state.index_ready:
    with st.spinner("🔧 Setting up RAG index..."):
        try:
            st.session_state.query_engine = setup_rag_index()
            st.session_state.index_ready = True
            st.success("✅ RAG index ready!")
        except Exception as e:
            st.error(f"❌ Failed to setup RAG index: {str(e)}")
            st.session_state.index_ready = False

# Chat Interface
st.subheader("💬 Chat Interface")

# Display chat messages in a container
messages_container = st.container()
with messages_container:
    if st.session_state.messages:
        for message in st.session_state.messages:
            if message["role"] == "user":
                with st.chat_message("user"):
                    st.write(message["content"])
            else:
                with st.chat_message("assistant"):
                    st.write(message["content"])
    else:
        st.info("No messages yet. Start by asking a question about your documents!")

# Chat input at bottom (outside the scrollable container)
if prompt := st.chat_input("Ask a question about your documents..."):
    # Add user message
    st.session_state.messages.append({"role": "user", "content": prompt})

    # Generate AI response
    with st.spinner("🤖 Thinking..."):
        try:
            if st.session_state.index_ready and st.session_state.query_engine:
                # Use actual RAG pipeline
                response = query_documents(
                    prompt, st.session_state.query_engine)
                response_text = str(response)
            else:
                available_docs = get_documents_in_folder()
                if available_docs:
                    response_text = f"⚠️ RAG index not ready yet. Found {len(available_docs)} document(s) in folder: {', '.join(available_docs)}. Please wait for the index to be set up or upload documents first."
                else:
                    response_text = "📁 No documents available. Please upload some documents first to enable RAG functionality."
        except Exception as e:
            response_text = f"❌ Error processing your question: {str(e)}"

    # Add AI response to session state
    st.session_state.messages.append(
        {"role": "assistant", "content": response_text})
    st.rerun()


show_code(setup_rag_index)

# Footer info
st.sidebar.divider()
st.sidebar.caption(
    "🚀 Powered by Heroku AI • Built with Streamlit • LlamaIndex RAG Pipeline")
