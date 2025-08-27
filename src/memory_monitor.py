#!/usr/bin/env python3
"""
Memory monitoring script to track system resources during ingestion.
Run this in a separate terminal while running the ingest process.
"""

import psutil
import time
import sys
import os
from datetime import datetime

def monitor_memory(interval=5, duration=None):
    """Monitor system memory usage and log to file"""
    
    # Create logs directory if it doesn't exist
    log_dir = os.path.join(os.path.dirname(__file__), "logs")
    os.makedirs(log_dir, exist_ok=True)
    
    log_file = os.path.join(log_dir, f"memory_monitor_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log")
    
    print(f"Starting memory monitoring (interval: {interval}s)")
    print(f"Logging to: {log_file}")
    print("Press Ctrl+C to stop")
    
    start_time = time.time()
    
    with open(log_file, 'w') as f:
        # Write header
        f.write("timestamp,elapsed_s,total_mb,available_mb,used_mb,used_percent,swap_used_mb,swap_percent\n")
        
        try:
            while True:
                current_time = time.time()
                elapsed = current_time - start_time
                
                # Get memory info
                memory = psutil.virtual_memory()
                swap = psutil.swap_memory()
                
                # Convert to MB
                total_mb = memory.total / 1024 / 1024
                available_mb = memory.available / 1024 / 1024
                used_mb = memory.used / 1024 / 1024
                swap_used_mb = swap.used / 1024 / 1024
                
                # Log data
                timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                log_line = f"{timestamp},{elapsed:.1f},{total_mb:.1f},{available_mb:.1f},{used_mb:.1f},{memory.percent:.1f},{swap_used_mb:.1f},{swap.percent:.1f}"
                
                f.write(log_line + "\n")
                f.flush()
                
                # Print to console
                print(f"[{timestamp}] Memory: {used_mb:.1f}MB/{total_mb:.1f}MB ({memory.percent:.1f}%) | "
                      f"Available: {available_mb:.1f}MB | Swap: {swap_used_mb:.1f}MB ({swap.percent:.1f}%)")
                
                # Check if duration limit reached
                if duration and elapsed >= duration:
                    print(f"Monitoring duration ({duration}s) reached. Stopping.")
                    break
                
                time.sleep(interval)
                
        except KeyboardInterrupt:
            print("\nMonitoring stopped by user")
        except Exception as e:
            print(f"Error during monitoring: {e}")
    
    print(f"Memory monitoring log saved to: {log_file}")

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Monitor system memory usage")
    parser.add_argument("--interval", "-i", type=int, default=5, help="Monitoring interval in seconds (default: 5)")
    parser.add_argument("--duration", "-d", type=int, help="Monitoring duration in seconds (default: unlimited)")
    
    args = parser.parse_args()
    
    monitor_memory(interval=args.interval, duration=args.duration)
