-- Initialize PostgreSQL database for Agentic RAG System
-- This script creates the necessary tables and extensions

-- Create the main database schema
CREATE SCHEMA IF NOT EXISTS agentic_rag;

-- Create documents table with embedding support
CREATE TABLE IF NOT EXISTS agentic_rag.documents (
    id SERIAL PRIMARY KEY,
    title TEXT NOT NULL,
    content TEXT NOT NULL,
    summary TEXT,
    metadata JSONB DEFAULT '{}',
    embedding TEXT, -- Store as JSON text for now
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

-- Create users table
CREATE TABLE IF NOT EXISTS agentic_rag.users (
    id TEXT PRIMARY KEY,
    name TEXT NOT NULL,
    email TEXT NOT NULL UNIQUE,
    description TEXT,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

-- Create indexes for better performance
CREATE INDEX IF NOT EXISTS idx_documents_title ON agentic_rag.documents USING GIN (to_tsvector('english', title));
CREATE INDEX IF NOT EXISTS idx_documents_content ON agentic_rag.documents USING GIN (to_tsvector('english', content));
CREATE INDEX IF NOT EXISTS idx_documents_created_at ON agentic_rag.documents (created_at DESC);
CREATE INDEX IF NOT EXISTS idx_users_email ON agentic_rag.users (email);
CREATE INDEX IF NOT EXISTS idx_users_created_at ON agentic_rag.users (created_at DESC);

-- Grant permissions (using DO block to handle potential errors gracefully)
DO $$
BEGIN
    -- Grant schema usage
    GRANT USAGE ON SCHEMA agentic_rag TO postgres;
    GRANT ALL PRIVILEGES ON ALL TABLES IN SCHEMA agentic_rag TO postgres;
    GRANT ALL PRIVILEGES ON ALL SEQUENCES IN SCHEMA agentic_rag TO postgres;
    
    -- Set default privileges for future objects
    ALTER DEFAULT PRIVILEGES IN SCHEMA agentic_rag GRANT ALL ON TABLES TO postgres;
    ALTER DEFAULT PRIVILEGES IN SCHEMA agentic_rag GRANT ALL ON SEQUENCES TO postgres;
EXCEPTION
    WHEN OTHERS THEN
        -- Log the error but continue (privileges may already exist)
        RAISE NOTICE 'Some privileges may already exist: %', SQLERRM;
END $$;
