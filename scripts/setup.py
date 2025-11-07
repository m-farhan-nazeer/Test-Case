#!/usr/bin/env python3
"""
Setup script for the Branch2 Agentic RAG system.
Automates the installation and configuration process for the multi-language architecture.

Copyright (c) 2025 Mike Tallent & Claude Sonnet - Geniusai.biz
All rights reserved. This software is proprietary and confidential.
"""

import os
import sys
import subprocess
import time
import shutil
from pathlib import Path

def run_command(command, cwd=None, check=True):
    """Run a shell command and return the result."""
    print(f"Running: {command}")
    try:
        result = subprocess.run(
            command, 
            shell=True, 
            cwd=cwd, 
            check=check,
            capture_output=True,
            text=True
        )
        if result.stdout:
            print(result.stdout)
        return result
    except subprocess.CalledProcessError as e:
        print(f"Error running command: {e}")
        if e.stderr:
            print(f"Error output: {e.stderr}")
        if check:
            sys.exit(1)
        return e

def check_prerequisites():
    """Check if required software is installed."""
    print("Checking prerequisites for Branch2 multi-language system...")
    
    required_tools = {
        "docker": "Docker",
        "docker-compose": "Docker Compose", 
        "node": "Node.js 18+",
        "npm": "npm",
        "go": "Go 1.21+",
        "java": "Java 17+",
        "cargo": "Rust/Cargo"
    }
    
    missing_tools = []
    
    for tool, description in required_tools.items():
        try:
            result = run_command(f"{tool} --version", check=False)
            if result.returncode != 0:
                missing_tools.append(description)
            else:
                print(f"✅ {description} is available")
        except:
            missing_tools.append(description)
    
    if missing_tools:
        print(f"\n❌ Missing required tools: {', '.join(missing_tools)}")
        print("\nPlease install the missing tools and run setup again.")
        print("\nInstallation guides:")
        print("- Docker: https://docs.docker.com/get-docker/")
        print("- Node.js: https://nodejs.org/")
        print("- Go: https://golang.org/doc/install")
        print("- Java: https://adoptium.net/")
        print("- Rust: https://rustup.rs/")
        sys.exit(1)
    
    print("✅ All prerequisites are available!")

def setup_environment_file():
    """Create the .env file if it doesn't exist."""
    print("\nSetting up environment configuration...")
    
    env_file = Path(".env")
    if env_file.exists():
        print(".env file already exists")
        return
    
    # Copy from example
    example_file = Path(".env.example")
    if example_file.exists():
        shutil.copy(example_file, env_file)
        print(".env file created from example")
        print("⚠️  Please edit .env file with your API keys (especially OPENAI_API_KEY)")
    else:
        print("❌ .env.example file not found")
        sys.exit(1)

def setup_frontend():
    """Set up the React frontend."""
    print("\nSetting up React frontend...")
    
    frontend_dir = Path("frontend")
    if not frontend_dir.exists():
        print("❌ Frontend directory not found")
        sys.exit(1)
    
    # Install npm dependencies
    print("📦 Installing frontend dependencies...")
    run_command("npm install", cwd=frontend_dir)
    
    print("✅ Frontend setup complete")

def setup_go_api():
    """Set up the Go API Gateway."""
    print("\nSetting up Go API Gateway...")
    
    go_api_dir = Path("branch2/go-api")
    if not go_api_dir.exists():
        print("❌ Go API directory not found")
        sys.exit(1)
    
    # Download Go dependencies
    print("📦 Installing Go dependencies...")
    run_command("go mod download", cwd=go_api_dir)
    
    print("✅ Go API setup complete")

def setup_java_services():
    """Set up the Java services."""
    print("\nSetting up Java services...")
    
    java_dir = Path("branch2/java-services")
    if not java_dir.exists():
        print("❌ Java services directory not found")
        sys.exit(1)
    
    # Download Maven dependencies
    print("📦 Installing Java dependencies...")
    if os.name == 'nt':  # Windows
        run_command("mvnw.cmd dependency:resolve", cwd=java_dir)
    else:  # Unix/Linux/Mac
        run_command("./mvnw dependency:resolve", cwd=java_dir)
    
    print("✅ Java services setup complete")

def setup_rust_core():
    """Set up the Rust core service."""
    print("\nSetting up Rust core service...")
    
    rust_dir = Path("branch2/rust-core")
    if not rust_dir.exists():
        print("❌ Rust core directory not found")
        sys.exit(1)
    
    # Build Rust dependencies
    print("📦 Building Rust dependencies...")
    run_command("cargo build", cwd=rust_dir)
    
    print("✅ Rust core setup complete")

def start_services():
    """Start Docker services."""
    print("\nStarting Docker services...")
    
    # Check if docker-compose.yml exists in branch2
    compose_file = Path("branch2/docker-compose.yml")
    if not compose_file.exists():
        print("❌ branch2/docker-compose.yml not found")
        sys.exit(1)
    
    # Start services
    run_command("docker-compose -f branch2/docker-compose.yml up -d")
    
    print("Waiting for services to be ready...")
    time.sleep(45)  # Wait longer for all services to start
    
    # Check service health
    services = ["postgres", "redis", "elasticsearch"]
    for service in services:
        result = run_command(f"docker-compose -f branch2/docker-compose.yml ps {service}", check=False)
        if "Up" in result.stdout:
            print(f"✅ {service} is running")
        else:
            print(f"⚠️  {service} may not be ready yet")
    
    print("✅ Docker services started")

def initialize_databases():
    """Initialize the databases with sample data."""
    print("\nInitializing databases...")
    
    # Install minimal Python dependencies for database initialization
    print("📦 Installing Python dependencies for database setup...")
    run_command("pip install psycopg2-binary redis elasticsearch python-dotenv requests")
    
    # Run database initialization script
    run_command("python scripts/init_databases.py")
    
    print("✅ Databases initialized")

def verify_setup():
    """Verify that the setup was successful."""
    print("\nVerifying setup...")
    
    # Check if services are responding
    try:
        import requests
        import time
        
        # Wait a bit more for services to be fully ready
        time.sleep(10)
        
        # Check Go API Gateway health
        try:
            response = requests.get("http://localhost:8000/health", timeout=5)
            if response.status_code == 200:
                print("✅ Go API Gateway is responding")
            else:
                print("⚠️  Go API Gateway may not be fully ready")
        except:
            print("⚠️  Go API Gateway is not responding yet")
        
        print("✅ Setup verification complete")
        
    except ImportError:
        print("⚠️  Cannot verify setup (requests module not available)")

def print_next_steps():
    """Print instructions for next steps."""
    print("\n" + "="*60)
    print("🎉 Branch2 Agentic RAG System Setup Complete!")
    print("="*60)
    
    print("\n📝 Next steps:")
    print("\n1. 🔑 Configure your API keys in .env:")
    print("   - Add your OPENAI_API_KEY")
    print("   - Update JWT_SECRET for production")
    
    print("\n2. 🚀 Start the system:")
    print("   Option A - Full Docker deployment:")
    print("     npm run docker:up")
    print("   ")
    print("   Option B - Development mode (with hot reload):")
    print("     npm run dev:full")
    
    print("\n3. 🌐 Access the application:")
    print("   - Frontend:        http://localhost:3000")
    print("   - Go API Gateway:  http://localhost:8000")
    print("   - API Health:      http://localhost:8000/health")
    print("   - Java Services:   http://localhost:8002")
    print("   - Rust Core:       http://localhost:8001 (gRPC)")
    
    print("\n4. 🧪 Test the system:")
    print("   - Open http://localhost:3000")
    print("   - Try asking a question")
    print("   - Upload a document")
    print("   - Check system stats")
    
    print("\n📚 Documentation:")
    print("   - Architecture: branch2/docs/architecture-diagram.md")
    print("   - API Reference: branch2/docs/api-reference.md")
    print("   - Development: branch2/docs/development-guide.md")
    
    print("\n🆘 Troubleshooting:")
    print("   - Check service logs: docker-compose -f branch2/docker-compose.yml logs")
    print("   - Restart services: docker-compose -f branch2/docker-compose.yml restart")
    print("   - View README.md for detailed instructions")

def main():
    """Main setup function."""
    print("Branch2 Agentic RAG System Setup")
    print("Multi-Language Architecture (Rust + Go + Java + TypeScript)")
    print("=" * 70)
    
    try:
        check_prerequisites()
        setup_environment_file()
        setup_frontend()
        setup_go_api()
        setup_java_services()
        setup_rust_core()
        start_services()
        initialize_databases()
        verify_setup()
        print_next_steps()
        
    except KeyboardInterrupt:
        print("\n❌ Setup interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Setup failed: {e}")
        print("\nPlease check the error messages above and try again.")
        print("If the issue persists, please check the documentation or create an issue.")
        sys.exit(1)

if __name__ == "__main__":
    main()
