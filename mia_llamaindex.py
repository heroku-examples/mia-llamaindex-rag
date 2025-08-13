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
from db import clear_vector_database, get_database_document_count, get_indexed_documents, delete_document_from_index

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
if 'uploader_key' not in st.session_state:
    st.session_state.uploader_key = 'file_uploader_1'

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

    # Load data from the documents folder and exclude already indexed documents
    # Already indexed: SELECT DISTINCT metadata_->>'file_name' as file_name FROM data_documents WHERE metadata_->>'file_name' IS NOT NULL
    already_indexed = get_indexed_documents()
    try:
        documents = SimpleDirectoryReader(
            DOCUMENTS_FOLDER, exclude=already_indexed).load_data()
    except Exception as e:
        documents = []

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

    # Handle different scenarios: new documents vs existing index
    if len(documents) == 0:
        # No new documents to index, load existing index from vector store
        index = VectorStoreIndex.from_vector_store(
            vector_store=vector_store,
            embed_model=embed_model,
            storage_context=storage_context
        )
    else:
        # Parse documents into nodes with proper chunking
        nodes = text_splitter.get_nodes_from_documents(documents)

        # Create/update the index with new nodes
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
        # "tree_summarize", "refine", "compact", "simple_summarize"
        response_mode=response_mode,
        similarity_top_k=top_k  # Number of relevant document chunks to retrieve
    )

    # Use the query_engine to query the documents
    # Example:
    # response = query_engine.query("What are the benefits of AI?")
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


def delete_document(filename):
    """Delete a document from the documents folder (with protection for certain files)"""
    # Protected files that cannot be deleted
    protected_files = ["AI and Machine Learning Overview.txt"]

    if filename in protected_files:
        return False, f"Cannot delete {filename} - this file is protected"

    try:
        file_path = DOCUMENTS_FOLDER / filename
        if file_path.exists():
            # Check if file is writable
            if not os.access(file_path, os.W_OK):
                return False, f"Permission denied: Cannot delete {filename}"

            file_path.unlink()

            # Verify file deletion was successful
            if file_path.exists():
                return False, f"Failed to delete {filename} - file still exists"

            # Also delete the document from the vector database
            db_success, db_message = delete_document_from_index(filename)
            if not db_success:
                # File was deleted but database cleanup failed - log this but don't fail the operation
                return True, f"File {filename} deleted, but database cleanup failed: {db_message}"

            return True, f"Successfully deleted {filename} from both file system and vector database"
        else:
            return False, f"File {filename} not found at path: {file_path}"
    except PermissionError:
        return False, f"Permission denied: Cannot delete {filename}"
    except Exception as e:
        return False, f"Error deleting {filename}: {str(e)}"


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
    # RAG Index Status Section
    st.subheader("📊 RAG Index Status")

    # Check if index is ready
    if st.session_state.index_ready:
        st.success("✅ RAG index ready!")
    else:
        st.warning("⚙️ RAG index is not ready.")

     # Automatic RAG Index Setup
    available_docs = get_documents_in_folder()
    if available_docs and not st.session_state.index_ready:
        with st.spinner("Setting up RAG index..."):
            try:
                st.session_state.query_engine = setup_rag_index()
                st.session_state.index_ready = True
                # Refresh UI to update sidebar status
                st.rerun()
            except Exception as e:
                st.error(f"❌ Failed to setup RAG index: {str(e)}")
                st.session_state.index_ready = False

    # Get document count from database
    doc_count, count_message = get_database_document_count()
    indexed_files = get_indexed_documents()

    if doc_count is not None:
        # Display document count
        if doc_count == 0:
            st.info("📄 **Documents in Index:** 0")
            st.caption("No documents indexed yet")
        else:
            st.success(f"📄 **Documents in Index:** {doc_count}")
            st.caption(
                f"{doc_count} document chunk(s) stored in vector database")

            # Show indexed files if any
            if indexed_files:
                with st.expander(f"📋 View Indexed Files ({len(indexed_files)})"):
                    for file in indexed_files:
                        st.text(f"📄 {file}")
    else:
        st.error("❌ **Database Status:** Error")
        st.caption(f"Could not connect: {count_message}")

    st.divider()

    # Reset Chat Button
    if st.button("🗑️ Reset", use_container_width=True):
        # Clear chat messages
        st.session_state.messages = []

        # Clear vector database
        with st.spinner("Clearing vector database..."):
            db_success, db_message = clear_vector_database()
            if db_success:
                st.success(db_message)
            else:
                st.error(db_message)

        # Reset index and query engine
        st.session_state.index_ready = False
        st.session_state.query_engine = None

        # Clear uploaded files tracking
        st.session_state.documents_uploaded = []

        # Clear file uploader by changing its key
        current_key = st.session_state.uploader_key
        if current_key == 'file_uploader_1':
            st.session_state.uploader_key = 'file_uploader_2'
        else:
            st.session_state.uploader_key = 'file_uploader_1'

        st.rerun()

# Main content area
st.title("🤖 RAG Chat with LlamaIndex and Heroku AI")

# Documents Overview Section
upload_container, documents_container = st.columns(2)

with upload_container:
    st.subheader("📁 Upload Documents")
    # Use a key that can be changed to clear the uploader
    uploader_key = st.session_state.get('uploader_key', 'file_uploader_1')
    uploaded_files = st.file_uploader(
        "Choose files to upload for RAG",
        accept_multiple_files=True,
        type=['txt', 'pdf', 'docx', 'md', 'csv'],
        key=uploader_key
    )

with documents_container:
    st.subheader("📋 Documents Folder")
    available_docs = get_documents_in_folder()
    if available_docs:
        st.info(f"Found {len(available_docs)} document(s) in folder")

        # Display each document with a delete button
        for doc in available_docs:
            col1, col2 = st.columns([4, 1])

            with col1:
                # Show protected status for certain files
                protected_files = ["AI and Machine Learning Overview.txt"]
                if doc in protected_files:
                    st.text(f"🔒 📄 {doc} (protected)")
                else:
                    st.text(f"📄 {doc}")

            with col2:
                # Only show delete button for non-protected files
                if doc not in protected_files:
                    if st.button("🗑️", key=f"delete_{doc}", help=f"Delete {doc}"):
                        success, message = delete_document(doc)
                        if success:
                            # Reset index status to trigger rebuild
                            st.session_state.index_ready = False
                            st.session_state.query_engine = None
                            # Remove from uploaded files list if it exists
                            if doc in st.session_state.documents_uploaded:
                                st.session_state.documents_uploaded.remove(doc)
                            # Clear file uploader by changing its key
                            current_key = st.session_state.uploader_key
                            if current_key == 'file_uploader_1':
                                st.session_state.uploader_key = 'file_uploader_2'
                            else:
                                st.session_state.uploader_key = 'file_uploader_1'
                            # Store success message in session state so it persists after rerun
                            st.session_state.delete_message = (
                                "success", message)
                            st.rerun()
                        else:
                            st.session_state.delete_message = (
                                "error", message)
                            st.rerun()

    else:
        st.warning("No documents in folder")

    # Display delete messages if any
    if hasattr(st.session_state, 'delete_message'):
        msg_type, msg_text = st.session_state.delete_message
        if msg_type == "success":
            st.success(msg_text)
        else:
            st.error(msg_text)
        # Clear the message after displaying
        del st.session_state.delete_message

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
if prompt := st.chat_input("Ask a question about your documents, for example: What are the benefits of AI?"):
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
