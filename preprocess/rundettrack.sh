#!/bin/bash

# Set log file
log_file="dettrack_log.txt"

# Add timestamp to log
log_with_time() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $1" | tee -a "$log_file"
}

# Check last line of log file
check_last_line() {
    if [ -f "$log_file" ]; then
        # Get last 10 lines (memory error message may not be the last line)
        last_lines=$(tail -n 10 "$log_file")
        if echo "$last_lines" | grep -q "GPU out of memory error"; then
            log_with_time "Found GPU out of memory error message"
            return 0  # Return 0 means memory error found, need to restart
        fi
    fi
    return 1  # Return 1 means no memory error found, no restart needed
}

# Check if truly completed
check_completion() {
    if [ -f "$log_file" ]; then
        last_lines=$(tail -n 10 "$log_file")
        # If last 10 lines contain both memory error and completion message, it's a false completion
        if echo "$last_lines" | grep -q "GPU out of memory error"; then
            if echo "$last_lines" | grep -q "All videos have been processed"; then
                return 1  # Return 1 means false completion
            fi
        fi
        # Only if the last line is completion message, it's true completion
        if tail -n 1 "$log_file" | grep -q "All videos have been processed"; then
            return 0  # Return 0 means truly completed
        fi
    fi
    return 1  # Return 1 means not completed
}

# Store all arguments passed to the script
SCRIPT_ARGS=("$@")

# Set error handling
set -e

while true; do
    # Check if last run terminated due to memory error
    if check_last_line; then
        log_with_time "Detected last run terminated due to GPU memory error, preparing to restart..."
        log_with_time "Waiting 30 seconds for GPU memory to be released..."
        sleep 30
        
        # Try to clear GPU memory
        log_with_time "Attempting to clear GPU memory..."
        nvidia-smi --gpu-reset 2>&1 | tee -a "$log_file" || true
    fi

    log_with_time "Starting dettrack.py with arguments: ${SCRIPT_ARGS[*]}"
    
    # Run Python script and record return value
    python dettrack.py "${SCRIPT_ARGS[@]}" 2>&1 | tee -a "$log_file"
    
    # Display current GPU status
    log_with_time "Current GPU status:"
    nvidia-smi | tee -a "$log_file" || true
    
    # First check if memory error occurred
    if check_last_line; then
        log_with_time "Detected GPU memory issue, preparing to restart program..."
        log_with_time "Waiting 30 seconds before restart..."
        sleep 30
        continue
    fi
    
    # Check again if truly completed
    if check_completion; then
        log_with_time "Confirmed all videos processed, exiting program"
        break
    else
        log_with_time "Continuing processing..."
        sleep 5
        continue
    fi
done
