#!/usr/bin/env python3
"""
Start all services for the Agentic RAG system.

This script helps start all backend services in the correct order
and provides status monitoring for the browser-host chatbot.
"""

import os
import sys
import time
import subprocess
import threading
import signal
from pathlib import Path
from dotenv import load_dotenv

# Handle Windows-specific imports
if os.name == 'nt':
    import subprocess

# Load environment variables
load_dotenv()

class ServiceManager:
    """Manages starting and monitoring of all services."""
    
    def __init__(self):
        self.processes = {}
        self.running = True
        
    def start_service(self, name, command, cwd=None, env=None):
        """Start a service with the given command."""
        print(f"[INFO] Starting {name}...")
        
        # Check if working directory exists
        if cwd and not os.path.exists(cwd):
            print(f"[ERROR] Working directory does not exist: {cwd}")
            return False
        
        try:
            # Set up environment
            if env is None:
                env = os.environ.copy()
            
            # Create log files for each service
            log_dir = os.path.join(os.path.dirname(__file__), '..', 'logs')
            os.makedirs(log_dir, exist_ok=True)
            
            stdout_file = open(os.path.join(log_dir, f"{name.lower().replace(' ', '_')}_stdout.log"), 'w')
            stderr_file = open(os.path.join(log_dir, f"{name.lower().replace(' ', '_')}_stderr.log"), 'w')
            
            # Start process without capturing output to avoid deadlock
            process = subprocess.Popen(
                command,
                cwd=cwd,
                env=env,
                stdout=stdout_file,
                stderr=stderr_file,
                shell=True,
                # Don't use CREATE_NEW_CONSOLE as it can cause hanging
                creationflags=0 if os.name == 'nt' else 0
            )
            
            self.processes[name] = {
                'process': process,
                'stdout_file': stdout_file,
                'stderr_file': stderr_file
            }
            
            print(f"[SUCCESS] {name} started with PID {process.pid}")
            print(f"[INFO] Logs: {log_dir}/{name.lower().replace(' ', '_')}_*.log")
            
            # Give the process a moment to start and check if it's still running
            time.sleep(2)
            if process.poll() is not None:
                print(f"[ERROR] {name} exited immediately with code {process.returncode}")
                # Read last few lines of stderr for debugging
                try:
                    stderr_file.flush()
                    with open(stderr_file.name, 'r') as f:
                        lines = f.readlines()
                        if lines:
                            print(f"[ERROR] {name} last error lines:")
                            for line in lines[-5:]:  # Show last 5 lines
                                print(f"  {line.strip()}")
                except Exception:
                    pass
                return False
            
            return True
            
        except Exception as e:
            print(f"[ERROR] Failed to start {name}: {e}")
            return False
    
    def monitor_services(self):
        """Monitor all running services."""
        while self.running:
            for name, proc_info in self.processes.items():
                process = proc_info['process']
                if process.poll() is not None:
                    print(f"[WARNING] {name} has stopped (exit code: {process.returncode})")
                    # Close log files
                    try:
                        proc_info['stdout_file'].close()
                        proc_info['stderr_file'].close()
                    except Exception:
                        pass
            
            time.sleep(5)
    
    def stop_all_services(self):
        """Stop all running services."""
        print("\n[INFO] Stopping all services...")
        self.running = False
        
        for name, proc_info in self.processes.items():
            process = proc_info['process']
            if process.poll() is None:
                print(f"[INFO] Stopping {name}...")
                process.terminate()
                
                # Wait for graceful shutdown
                try:
                    process.wait(timeout=10)
                    print(f"[SUCCESS] {name} stopped gracefully")
                except subprocess.TimeoutExpired:
                    print(f"[WARNING] Force killing {name}...")
                    process.kill()
            
            # Close log files
            try:
                proc_info['stdout_file'].close()
                proc_info['stderr_file'].close()
            except Exception:
                pass
    
    def signal_handler(self, signum, frame):
        """Handle interrupt signals."""
        print(f"\n[INFO] Received signal {signum}")
        self.stop_all_services()
        sys.exit(0)

def check_dependencies():
    """Check if required tools are installed."""
    print("[INFO] Checking dependencies...")
    
    dependencies = {
        'python': ['python', '--version'],
        'uvicorn': ['uvicorn', '--version'],
        'go': ['go', 'version'],
        'mvn': ['mvn', '--version'],
        'cargo': ['cargo', '--version']
    }
    
    missing = []
    for tool, cmd in dependencies.items():
        try:
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=5)
            if result.returncode == 0:
                print(f"[SUCCESS] {tool} is available")
            else:
                missing.append(tool)
                print(f"[WARNING] {tool} check failed")
        except (subprocess.TimeoutExpired, FileNotFoundError):
            missing.append(tool)
            print(f"[WARNING] {tool} not found")
    
    if missing:
        print(f"[WARNING] Missing tools: {', '.join(missing)}")
        print("[INFO] Some services may not start")
    
    return len(missing) == 0

def main():
    """Main function to start all services."""
    print("=== Agentic RAG System Service Manager ===")
    print("Starting all backend services for browser-host chatbot integration")
    print()
    
    # Check dependencies first
    check_dependencies()
    print()
    
    # Get project root directory
    project_root = Path(__file__).parent.parent
    print(f"[INFO] Project root: {project_root}")
    
    # Check if required directories exist
    required_dirs = [
        project_root / "api",
        project_root / "branch2" / "go-api", 
        project_root / "branch2" / "java-services",
        project_root / "branch2" / "rust-core"
    ]
    
    missing_dirs = [d for d in required_dirs if not d.exists()]
    if missing_dirs:
        print("[WARNING] Some service directories are missing:")
        for d in missing_dirs:
            print(f"  - {d}")
        print("[INFO] Will skip missing services")
    
    # Initialize service manager
    manager = ServiceManager()
    
    # Set up signal handlers
    signal.signal(signal.SIGINT, manager.signal_handler)
    if hasattr(signal, 'SIGTERM'):  # SIGTERM not available on Windows
        signal.signal(signal.SIGTERM, manager.signal_handler)
    
    # Start services in order with better commands
    services = [
        {
            "name": "Python FastAPI",
            "command": "python -m uvicorn api.main:app --reload --port 8000 --host 0.0.0.0",
            "cwd": str(project_root)
        },
        {
            "name": "Go API",
            "command": "go run main.go",
            "cwd": str(project_root / "branch2" / "go-api")
        },
        {
            "name": "Java Services", 
            "command": "mvn spring-boot:run -Dspring-boot.run.jvmArguments=\"-Xmx512m\"",
            "cwd": str(project_root / "branch2" / "java-services")
        },
        {
            "name": "Rust Core",
            "command": "cargo run --release",
            "cwd": str(project_root / "branch2" / "rust-core")
        }
    ]
    
    # Start each service
    started_services = []
    for service in services:
        # Check if service directory exists before trying to start
        if not os.path.exists(service['cwd']):
            print(f"[WARNING] Skipping {service['name']} - directory not found: {service['cwd']}")
            continue
            
        print(f"[INFO] Attempting to start {service['name']}...")
        if manager.start_service(**service):
            started_services.append(service['name'])
            print(f"[SUCCESS] {service['name']} started successfully")
        else:
            print(f"[ERROR] Failed to start {service['name']}, continuing with other services...")
        
        # Wait between service starts
        print(f"[INFO] Waiting 5 seconds before starting next service...")
        time.sleep(5)
    
    if started_services:
        print(f"\n[SUCCESS] Started {len(started_services)} services: {', '.join(started_services)}")
        print("[INFO] You can now open browser-host/index.html to use the chatbot")
        print("[INFO] Service logs are in the 'logs' directory")
        print("[INFO] Press Ctrl+C to stop all services")
        print()
    else:
        print("\n[ERROR] No services were started successfully!")
        print("[INFO] Check the logs directory for error details")
        return
    
    # Start monitoring thread
    monitor_thread = threading.Thread(target=manager.monitor_services)
    monitor_thread.daemon = True
    monitor_thread.start()
    
    # Keep main thread alive
    try:
        while manager.running:
            time.sleep(1)
    except KeyboardInterrupt:
        manager.signal_handler(signal.SIGINT, None)

if __name__ == "__main__":
    main()
