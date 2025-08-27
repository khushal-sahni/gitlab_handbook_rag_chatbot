#!/usr/bin/env python3
"""
Debug script to run the ingest process with comprehensive monitoring.
This script helps identify memory issues and bottlenecks.
"""

import os
import sys
import subprocess
import time
from datetime import datetime

def check_dependencies():
    """Check if required packages are installed"""
    try:
        import psutil
        import memory_profiler
        print("✓ Required packages (psutil, memory-profiler) are available")
        return True
    except ImportError as e:
        print(f"✗ Missing required package: {e}")
        print("Please install with: pip install psutil memory-profiler")
        return False

def get_system_info():
    """Get system information for debugging context"""
    try:
        import psutil
        
        # Memory info
        memory = psutil.virtual_memory()
        swap = psutil.swap_memory()
        
        print("\n=== SYSTEM INFORMATION ===")
        print(f"Total RAM: {memory.total / 1024 / 1024 / 1024:.1f} GB")
        print(f"Available RAM: {memory.available / 1024 / 1024 / 1024:.1f} GB")
        print(f"Used RAM: {memory.percent:.1f}%")
        print(f"Swap Total: {swap.total / 1024 / 1024 / 1024:.1f} GB")
        print(f"Swap Used: {swap.percent:.1f}%")
        
        # CPU info
        print(f"CPU Count: {psutil.cpu_count()}")
        print(f"CPU Usage: {psutil.cpu_percent(interval=1):.1f}%")
        
        return True
    except Exception as e:
        print(f"Error getting system info: {e}")
        return False

def run_debug_ingest():
    """Run the ingest process with debugging enabled"""
    
    print("=== GITLAB HANDBOOK RAG INGEST DEBUGGER ===")
    print(f"Started at: {datetime.now()}")
    
    # Check dependencies
    if not check_dependencies():
        return False
    
    # Get system info
    get_system_info()
    
    # Check if .env file exists
    env_file = ".env"
    if not os.path.exists(env_file):
        print(f"\n⚠️  Warning: {env_file} not found. Make sure you have configured your API keys.")
        print("Copy env.example to .env and fill in your API keys.")
    
    # Create logs directory
    logs_dir = os.path.join("src", "logs")
    os.makedirs(logs_dir, exist_ok=True)
    
    print(f"\n=== STARTING INGEST PROCESS ===")
    print(f"Logs will be saved to: {logs_dir}/ingest_debug.log")
    print(f"Memory profiling will be saved to: ingest_memory_profile.log")
    print("\nMonitoring memory usage during ingestion...")
    print("If the process gets killed, check the logs for the last operation before failure.\n")
    
    try:
        # Run the ingest process with memory profiling
        cmd = [
            sys.executable, "-m", "memory_profiler", 
            "-m", "src.ingest"
        ]
        
        # Set environment variable for memory profiling output
        env = os.environ.copy()
        env["MEMORY_PROFILER_OUTPUT"] = "ingest_memory_profile.log"
        
        # Run the process
        result = subprocess.run(cmd, env=env, capture_output=False, text=True)
        
        if result.returncode == 0:
            print("\n✓ Ingest process completed successfully!")
        elif result.returncode == -9:  # SIGKILL
            print("\n✗ Process was killed (likely due to memory issues)")
            print("Check the logs to see what operation was running when it failed.")
        else:
            print(f"\n✗ Process failed with return code: {result.returncode}")
        
        return result.returncode == 0
        
    except KeyboardInterrupt:
        print("\n⚠️  Process interrupted by user")
        return False
    except Exception as e:
        print(f"\n✗ Error running ingest process: {e}")
        return False

def analyze_logs():
    """Analyze the debug logs to identify issues"""
    logs_dir = os.path.join("src", "logs")
    log_file = os.path.join(logs_dir, "ingest_debug.log")
    
    if not os.path.exists(log_file):
        print(f"Log file not found: {log_file}")
        return
    
    print(f"\n=== LOG ANALYSIS ===")
    print(f"Analyzing: {log_file}")
    
    try:
        with open(log_file, 'r') as f:
            lines = f.readlines()
        
        # Look for memory warnings and errors
        memory_warnings = []
        errors = []
        last_operations = []
        
        for i, line in enumerate(lines):
            if "HIGH MEMORY USAGE" in line:
                memory_warnings.append((i+1, line.strip()))
            elif "ERROR" in line or "Critical error" in line:
                errors.append((i+1, line.strip()))
            elif "MEMORY [" in line:
                last_operations.append((i+1, line.strip()))
        
        print(f"\nMemory warnings found: {len(memory_warnings)}")
        for line_num, warning in memory_warnings[-5:]:  # Show last 5
            print(f"  Line {line_num}: {warning}")
        
        print(f"\nErrors found: {len(errors)}")
        for line_num, error in errors:
            print(f"  Line {line_num}: {error}")
        
        print(f"\nLast few memory operations:")
        for line_num, op in last_operations[-10:]:  # Show last 10
            print(f"  Line {line_num}: {op}")
            
    except Exception as e:
        print(f"Error analyzing logs: {e}")

if __name__ == "__main__":
    success = run_debug_ingest()
    
    # Always try to analyze logs, even if process failed
    analyze_logs()
    
    if not success:
        print("\n=== DEBUGGING TIPS ===")
        print("1. Check the memory usage patterns in the logs")
        print("2. Look for the last operation before failure")
        print("3. Consider reducing MAX_PAGES or batch size if memory is limited")
        print("4. Ensure you have enough available RAM (recommend 8GB+)")
        print("5. Close other memory-intensive applications")
        
    print(f"\nDebugging session completed at: {datetime.now()}")
