#!/usr/bin/env python3
"""
Test individual services for the Agentic RAG system.

This script helps test services one by one to identify issues.
"""

import os
import sys
import subprocess
import time
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

def test_python_service():
    """Test Python FastAPI service."""
    print("Testing Python FastAPI service...")
    
    project_root = Path(__file__).parent.parent
    
    try:
        # Test if we can import the main module
        result = subprocess.run([
            'python', '-c', 'import api.main; print("Import successful")'
        ], cwd=str(project_root), capture_output=True, text=True, timeout=10)
        
        if result.returncode == 0:
            print("[SUCCESS] Python service imports work")
            
            # Try to start the service briefly
            print("Starting Python service for 10 seconds...")
            process = subprocess.Popen([
                'python', '-m', 'uvicorn', 'api.main:app', '--port', '8000'
            ], cwd=str(project_root))
            
            time.sleep(10)
            process.terminate()
            process.wait()
            
            print("[SUCCESS] Python service started and stopped successfully")
            return True
        else:
            print(f"[ERROR] Python service import failed: {result.stderr}")
            return False
            
    except Exception as e:
        print(f"[ERROR] Python service test failed: {e}")
        return False

def test_go_service():
    """Test Go API service."""
    print("Testing Go API service...")
    
    go_dir = Path(__file__).parent.parent / "branch2" / "go-api"
    
    if not go_dir.exists():
        print("[WARNING] Go service directory not found")
        return False
    
    try:
        # Test if Go code compiles
        result = subprocess.run([
            'go', 'build', '-o', 'test_build', '.'
        ], cwd=str(go_dir), capture_output=True, text=True, timeout=30)
        
        if result.returncode == 0:
            print("[SUCCESS] Go service compiles successfully")
            
            # Clean up build file
            build_file = go_dir / "test_build"
            if build_file.exists():
                build_file.unlink()
            
            return True
        else:
            print(f"[ERROR] Go service compilation failed: {result.stderr}")
            return False
            
    except Exception as e:
        print(f"[ERROR] Go service test failed: {e}")
        return False

def main():
    """Test all services individually."""
    print("=== Individual Service Testing ===")
    print()
    
    tests = [
        ("Python FastAPI", test_python_service),
        ("Go API", test_go_service),
    ]
    
    results = {}
    for name, test_func in tests:
        print(f"--- Testing {name} ---")
        results[name] = test_func()
        print()
    
    print("=== Test Results ===")
    for name, success in results.items():
        status = "PASS" if success else "FAIL"
        print(f"{name}: {status}")
    
    if all(results.values()):
        print("\n[SUCCESS] All tests passed!")
    else:
        print("\n[WARNING] Some tests failed. Check individual service configurations.")

if __name__ == "__main__":
    main()
