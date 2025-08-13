"""
Database operations for the LlamaIndex RAG application.
Handles PostgreSQL operations for document storage and retrieval.
"""

import os
from sqlalchemy import create_engine, text
from dotenv import load_dotenv

load_dotenv()


def get_db_engine():
    """Get database engine for PostgreSQL operations"""
    try:
        # Get database connection string
        database_url = os.environ.get("DATABASE_URL")
        if not database_url:
            raise ValueError("DATABASE_URL not found in environment variables")

        # Convert postgres:// to postgresql:// for SQLAlchemy
        database_url = database_url.replace("postgres://", "postgresql://")

        # Create and return engine
        return create_engine(database_url)

    except Exception as e:
        raise Exception(f"Failed to create database engine: {str(e)}")


def clear_vector_database():
    """Clear all documents from the PostgreSQL vector database table"""
    try:
        engine = get_db_engine()

        with engine.connect() as connection:
            # Clear the documents table (this is the table name used in PGVectorStore)
            connection.execute(text("DELETE FROM data_documents"))
            connection.commit()

        return True, "Successfully cleared vector database"

    except Exception as e:
        return False, f"Error clearing vector database: {str(e)}"


def get_database_document_count():
    """Get the number of documents stored in the PostgreSQL vector database"""
    try:
        engine = get_db_engine()

        with engine.connect() as connection:
            # Count documents in the data_documents table
            result = connection.execute(
                text("SELECT COUNT(*) FROM data_documents"))
            count = result.scalar()

        return count, "Success"

    except Exception as e:
        return None, f"Error querying database: {str(e)}"


def get_indexed_documents():
    """Get list of document filenames that are already indexed in the database"""
    try:
        engine = get_db_engine()

        with engine.connect() as connection:
            # Query for distinct filenames from metadata
            result = connection.execute(
                text("SELECT DISTINCT metadata_->>'file_name' as file_name FROM data_documents WHERE metadata_->>'file_name' IS NOT NULL")
            )
            # Extract filenames from the result
            indexed_files = [row[0] for row in result.fetchall() if row[0]]

        return indexed_files

    except Exception as e:
        # If there's any error, return empty list to allow normal processing
        return []


def delete_document_from_index(filename):
    """Delete all document chunks from the vector database for a specific filename"""
    try:
        engine = get_db_engine()

        with engine.connect() as connection:
            # Delete all rows where the metadata contains the specific filename
            result = connection.execute(
                text(
                    "DELETE FROM data_documents WHERE metadata_->>'file_name' = :filename"),
                {"filename": filename}
            )
            deleted_count = result.rowcount
            connection.commit()

        return True, f"Successfully deleted {deleted_count} chunk(s) for {filename} from vector database"

    except Exception as e:
        return False, f"Error deleting {filename} from vector database: {str(e)}"
