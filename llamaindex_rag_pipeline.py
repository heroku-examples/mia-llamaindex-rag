from llama_index.llms.heroku import Heroku
from llama_index.embeddings.openai_like import OpenAILikeEmbedding
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader, Settings, StorageContext
from llama_index.core.node_parser import SentenceSplitter
from llama_index.vector_stores.postgres import PGVectorStore
import os
from sqlalchemy import make_url
from pathlib import Path

DOCUMENTS_FOLDER = Path("documents")
DOCUMENTS_FOLDER.mkdir(exist_ok=True)  # Ensure folder exists

# LlamaIndex RAG Pipeline
# response_mode: "tree_summarize", "refine", "compact", "simple_summarize"
# top_k: Number of relevant document chunks to retrieve
# indexed_files: List of files that are already indexed - SELECT DISTINCT metadata_->>'file_name' as file_name FROM data_documents WHERE metadata_->>'file_name' IS NOT NULL


def setup_rag_index(response_mode="tree_summarize", top_k=10, indexed_files=[]):
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
    try:
        documents = SimpleDirectoryReader(
            DOCUMENTS_FOLDER, exclude=indexed_files).load_data()
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

    return query_engine


def query_documents(prompt, query_engine):
    """Query the documents using the pre-built query engine"""
    response = query_engine.query(prompt)
    return response
