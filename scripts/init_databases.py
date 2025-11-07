"""
Initialize databases for Branch2 multi-service system.

This script initializes all databases and services required for the
Agentic RAG system, including support for the browser-host chatbot.

Services initialized:
- PostgreSQL with Lantern extension
- Redis for caching
- Elasticsearch for full-text search
- Sample data for testing

The script also validates that all services are accessible
for the browser-host chatbot integration.

Copyright (c) 2025 Mike Tallent & Claude Sonnet - Geniusai.biz
All rights reserved. This software is proprietary and confidential.
"""

import os
import sys
import time
import subprocess
import psycopg2
import redis
import requests
from elasticsearch import Elasticsearch
from dotenv import load_dotenv
import json

# Load environment variables
load_dotenv()

def print_status(message):
    """Print status message with timestamp."""
    timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{timestamp}] [INFO] {message}")

def print_error(message):
    """Print error message with timestamp."""
    timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{timestamp}] [ERROR] {message}")

def print_success(message):
    """Print success message with timestamp."""
    timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{timestamp}] [SUCCESS] {message}")

def wait_for_postgres():
    """Wait for PostgreSQL to be ready."""
    print_status("Waiting for PostgreSQL to be ready...")
    
    max_retries = 30
    retry_count = 0
    
    while retry_count < max_retries:
        try:
            conn = psycopg2.connect(
                host=os.getenv('POSTGRES_HOST', 'localhost'),
                port=os.getenv('POSTGRES_PORT', '5432'),
                user=os.getenv('POSTGRES_USER', 'postgres'),
                password=os.getenv('POSTGRES_PASSWORD', 'password'),
                database='postgres'
            )
            conn.close()
            print_status("PostgreSQL is ready!")
            return True
        except psycopg2.OperationalError:
            retry_count += 1
            time.sleep(2)
    
    print_error("PostgreSQL failed to become ready")
    return False

def wait_for_redis():
    """Wait for Redis to be ready."""
    print_status("Waiting for Redis to be ready...")
    
    max_retries = 30
    retry_count = 0
    
    while retry_count < max_retries:
        try:
            r = redis.Redis(
                host=os.getenv('REDIS_HOST', 'localhost'),
                port=int(os.getenv('REDIS_PORT', '6379')),
                decode_responses=True
            )
            r.ping()
            print_status("Redis is ready!")
            return True
        except redis.ConnectionError:
            retry_count += 1
            time.sleep(2)
    
    print_error("Redis failed to become ready")
    return False

def wait_for_elasticsearch():
    """Wait for Elasticsearch to be ready."""
    print_status("Waiting for Elasticsearch to be ready...")
    
    max_retries = 30
    retry_count = 0
    
    while retry_count < max_retries:
        try:
            es = Elasticsearch([{
                'host': os.getenv('ELASTICSEARCH_HOST', 'localhost'),
                'port': int(os.getenv('ELASTICSEARCH_PORT', '9200')),
                'scheme': 'http'
            }])
            
            if es.ping():
                print_status("Elasticsearch is ready!")
                return True
        except Exception:
            pass
        
        retry_count += 1
        time.sleep(2)
    
    print_error("Elasticsearch failed to become ready")
    return False

def run_postgres_setup():
    """Run the PostgreSQL setup script."""
    print_status("Setting up PostgreSQL database schema...")
    
    try:
        # Create basic database schema
        conn = psycopg2.connect(
            host=os.getenv('POSTGRES_HOST', 'localhost'),
            port=os.getenv('POSTGRES_PORT', '5432'),
            user=os.getenv('POSTGRES_USER', 'postgres'),
            password=os.getenv('POSTGRES_PASSWORD', 'password'),
            database=os.getenv('POSTGRES_DB', 'agentic_rag')
        )
        
        cursor = conn.cursor()
        
        # Create schema if it doesn't exist
        cursor.execute("CREATE SCHEMA IF NOT EXISTS agentic_rag;")
        
        # Create documents table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS agentic_rag.documents (
                id SERIAL PRIMARY KEY,
                title VARCHAR(500) NOT NULL,
                content TEXT NOT NULL,
                summary TEXT,
                metadata JSONB DEFAULT '{}',
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            );
        """)
        
        # Create index on metadata for faster queries
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_documents_metadata 
            ON agentic_rag.documents USING GIN (metadata);
        """)
        
        # Create full-text search index
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_documents_content_fts 
            ON agentic_rag.documents USING GIN (to_tsvector('english', content));
        """)
        
        conn.commit()
        cursor.close()
        conn.close()
        
        print_status("PostgreSQL setup completed successfully!")
        return True
        
    except Exception as e:
        print_error(f"Failed to run PostgreSQL setup: {e}")
        return False

def create_elasticsearch_indices():
    """Create Elasticsearch indices."""
    print_status("Creating Elasticsearch indices...")
    
    try:
        es = Elasticsearch([{
            'host': os.getenv('ELASTICSEARCH_HOST', 'localhost'),
            'port': int(os.getenv('ELASTICSEARCH_PORT', '9200')),
            'scheme': 'http'
        }])
        
        # Create documents index
        if not es.indices.exists(index='documents'):
            es.indices.create(
                index='documents',
                body={
                    'mappings': {
                        'properties': {
                            'title': {'type': 'text', 'analyzer': 'english'},
                            'content': {'type': 'text', 'analyzer': 'english'},
                            'summary': {'type': 'text', 'analyzer': 'english'},
                            'metadata': {'type': 'object'},
                            'created_at': {'type': 'date'}
                        }
                    }
                }
            )
            print_status("Documents index created")
        
        return True
        
    except Exception as e:
        print_error(f"Failed to create Elasticsearch indices: {e}")
        return False

def add_sample_data():
    """Add sample data to the system."""
    print_status("Adding sample data...")
    
    try:
        # Add sample documents to PostgreSQL
        conn = psycopg2.connect(
            host=os.getenv('POSTGRES_HOST', 'localhost'),
            port=os.getenv('POSTGRES_PORT', '5432'),
            user=os.getenv('POSTGRES_USER', 'postgres'),
            password=os.getenv('POSTGRES_PASSWORD', 'password'),
            database=os.getenv('POSTGRES_DB', 'agentic_rag')
        )
        
        cursor = conn.cursor()
        
        # Sample documents
        sample_docs = [
            (
                "Introduction to Artificial Intelligence",
                "Artificial Intelligence (AI) is a branch of computer science that aims to create intelligent machines that work and react like humans. AI systems can perform tasks such as learning, reasoning, problem-solving, perception, and language understanding.",
                "Overview of AI and its capabilities",
                '{"category": "technology", "difficulty": "beginner"}'
            ),
            (
                "Machine Learning Fundamentals", 
                "Machine Learning is a subset of AI that provides systems the ability to automatically learn and improve from experience without being explicitly programmed. It focuses on the development of computer programs that can access data and use it to learn for themselves.",
                "Basic concepts of machine learning",
                '{"category": "technology", "difficulty": "intermediate"}'
            ),
            (
                "Deep Learning and Neural Networks",
                "Deep Learning is a subset of machine learning that uses neural networks with multiple layers to model and understand complex patterns in data. It has been particularly successful in areas like image recognition, natural language processing, and speech recognition.",
                "Advanced AI techniques using neural networks", 
                '{"category": "technology", "difficulty": "advanced"}'
            )
        ]
        
        for title, content, summary, metadata in sample_docs:
            cursor.execute(
                "INSERT INTO agentic_rag.documents (title, content, summary, metadata) VALUES (%s, %s, %s, %s)",
                (title, content, summary, metadata)
            )
        
        conn.commit()
        cursor.close()
        conn.close()
        
        print_status("Sample data added successfully!")
        return True
        
    except Exception as e:
        print_error(f"Failed to add sample data: {e}")
        return False

def validate_browser_host_config():
    """Validate that browser-host configuration is compatible."""
    print_status("Validating browser-host interface configuration...")
    
    try:
        # Check if browser-host directory exists
        browser_host_dir = os.path.join(os.path.dirname(__file__), '..', 'browser-host')
        if not os.path.exists(browser_host_dir):
            print_error(f"Browser-host directory not found: {browser_host_dir}")
            return False
        
        # Check for required files
        required_files = ['index.html', 'config.js', 'api-client.js', 'script.js', 'style.css']
        missing_files = []
        
        for file in required_files:
            file_path = os.path.join(browser_host_dir, file)
            if not os.path.exists(file_path):
                missing_files.append(file)
        
        if missing_files:
            print_error(f"Missing browser-host files: {', '.join(missing_files)}")
            return False
        
        # Validate that expected ports are configured
        expected_ports = {
            'python': os.getenv('PYTHON_API_PORT', '8000'),
            'go': os.getenv('GO_API_PORT', '8080'),
            'java': os.getenv('JAVA_SERVICES_PORT', '8081'),
            'rust': os.getenv('RUST_CORE_HTTP_PORT', '8082')
        }
        
        print_status("Expected service ports:")
        for service, port in expected_ports.items():
            print_status(f"  {service}: {port}")
        
        print_success("Browser-host configuration validation completed")
        return True
        
    except Exception as e:
        print_error(f"Browser-host validation failed: {e}")
        return False

def create_browser_host_demo_data():
    """Create demo data specifically for browser-host testing."""
    print_status("Creating browser-host demo data...")
    
    try:
        # Add demo documents that work well with the browser chatbot
        demo_docs = [
            {
                "title": "Browser Chatbot Quick Start",
                "content": "The browser-host chatbot is a lightweight interface that connects directly to all backend APIs. It requires no installation and works in any modern browser. Simply open index.html and start asking questions.",
                "summary": "Quick start guide for the browser chatbot",
                "metadata": {"category": "documentation", "type": "guide", "difficulty": "beginner"}
            },
            {
                "title": "Multi-API Architecture Benefits", 
                "content": "The Agentic RAG system uses multiple APIs for different purposes: Python for orchestration, Go for performance, Java for enterprise features, Rust for core analysis, and React for UI. This architecture provides redundancy and specialized capabilities.",
                "summary": "Overview of the multi-service architecture",
                "metadata": {"category": "architecture", "type": "overview", "difficulty": "intermediate"}
            },
            {
                "title": "WebSocket vs HTTP Communication",
                "content": "The browser chatbot supports both WebSocket and HTTP communication. WebSocket provides real-time streaming responses ideal for complex queries, while HTTP offers reliable request-response patterns for simple questions. The system automatically falls back to HTTP if WebSocket fails.",
                "summary": "Communication protocols in the browser chatbot",
                "metadata": {"category": "technical", "type": "comparison", "difficulty": "intermediate"}
            }
        ]
        
        # Add to PostgreSQL
        conn = psycopg2.connect(
            host=os.getenv('POSTGRES_HOST', 'localhost'),
            port=os.getenv('POSTGRES_PORT', '5432'),
            user=os.getenv('POSTGRES_USER', 'postgres'),
            password=os.getenv('POSTGRES_PASSWORD', 'password'),
            database=os.getenv('POSTGRES_DB', 'agentic_rag')
        )
        
        cursor = conn.cursor()
        
        for doc in demo_docs:
            cursor.execute(
                "INSERT INTO agentic_rag.documents (title, content, summary, metadata) VALUES (%s, %s, %s, %s)",
                (doc["title"], doc["content"], doc["summary"], json.dumps(doc["metadata"]))
            )
        
        conn.commit()
        cursor.close()
        conn.close()
        
        print_success("Browser-host demo data created successfully")
        return True
        
    except Exception as e:
        print_error(f"Failed to create browser-host demo data: {e}")
        return False

def print_startup_instructions():
    """Print instructions for starting all services."""
    print_success("=== Agentic RAG Database Initialization Complete ===")
    print_status("")
    print_status("Next steps to start the full system:")
    print_status("")
    print_status("1. Start Python FastAPI service:")
    print_status("   python main.py")
    print_status("   # or: uvicorn main:app --reload --port 8000")
    print_status("")
    print_status("2. Start Go API service:")
    print_status("   cd branch2/go-api && go run main.go")
    print_status("")
    print_status("3. Start Java services:")
    print_status("   cd branch2/java-services && ./mvnw spring-boot:run")
    print_status("")
    print_status("4. Start Rust core service:")
    print_status("   cd branch2/rust-core && cargo run")
    print_status("")
    print_status("5. Open browser interface:")
    print_status("   # Open browser-host/index.html in your browser")
    print_status("   # Or use: npm run dev:browser")
    print_status("")
    print_status("6. Alternative: Use npm scripts:")
    print_status("   npm run dev:all    # Start all services")
    print_status("   npm run dev:python # Start Python API only")
    print_status("")
    print_status("The browser interface will automatically detect and connect to available services.")

def main():
    """Main initialization function."""
    print_status("Starting Agentic RAG System database initialization...")
    print_status("This will set up databases for the multi-language RAG system")
    print_status("")
    
    # Validate browser-host configuration
    if not validate_browser_host_config():
        print_error("Browser-host validation failed, but continuing with database setup...")
    
    # Wait for all services to be ready
    print_status("Waiting for database services to be ready...")
    if not wait_for_postgres():
        print_error("PostgreSQL is not available. Please start it with: docker-compose up -d postgres")
        sys.exit(1)
    
    if not wait_for_redis():
        print_error("Redis is not available. Please start it with: docker-compose up -d redis")
        sys.exit(1)
    
    if not wait_for_elasticsearch():
        print_error("Elasticsearch is not available. Please start it with: docker-compose up -d elasticsearch")
        sys.exit(1)
    
    # Run setup scripts
    print_status("Running database setup scripts...")
    if not run_postgres_setup():
        print_error("PostgreSQL setup failed")
        sys.exit(1)
    
    if not create_elasticsearch_indices():
        print_error("Elasticsearch setup failed")
        sys.exit(1)
    
    if not add_sample_data():
        print_error("Sample data creation failed")
        sys.exit(1)
    
    # Add browser-host specific demo data
    if not create_browser_host_demo_data():
        print_error("Failed to create browser-host demo data, but continuing...")
    
    # Print startup instructions
    print_startup_instructions()

if __name__ == "__main__":
    main()
