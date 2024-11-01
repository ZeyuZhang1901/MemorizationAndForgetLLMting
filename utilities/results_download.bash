#!/bin/bash

# Usage: ./results_download.bash <experiment_name> <remote_host>
# 1. make the file executable: chmod +x utilities/results_download.bash
# 2. configure the remaining variables below
# 3. run the script: ./results_download.bash <experiment_name> <remote_host>
# An example: ./results_download.bash name_number_query_sft 10.10.10.10

# Configuration
REMOTE_USER="ubuntu"
LOCAL_BASE_DIR="./downloaded_results"
REMOTE_BASE_DIR="/path/to/your/project"

# Create local directories if they don't exist
mkdir -p "${LOCAL_BASE_DIR}/logs"
mkdir -p "${LOCAL_BASE_DIR}/models"
mkdir -p "${LOCAL_BASE_DIR}/plots"

# Function to download experiment results
download_experiment_results() {
    local EXP_NAME=$1
    local REMOTE_HOST=$2
    local TIMESTAMP=$(date +%Y%m%d_%H%M%S)
    local LOCAL_EXP_DIR="${LOCAL_BASE_DIR}/${EXP_NAME}_${TIMESTAMP}"
    
    echo "Downloading results for experiment: ${EXP_NAME}"
    echo "From remote host: ${REMOTE_HOST}"
    
    # Create experiment-specific directory
    mkdir -p "${LOCAL_EXP_DIR}"
    
    # Download logs
    echo "Downloading logs..."
    scp -r "${REMOTE_USER}@${REMOTE_HOST}:${REMOTE_BASE_DIR}/logs/run_*" "${LOCAL_EXP_DIR}/logs/"
    
    # Download training plots
    echo "Downloading plots..."
    scp -r "${REMOTE_USER}@${REMOTE_HOST}:${REMOTE_BASE_DIR}/logs/run_*/training_progress.png" "${LOCAL_EXP_DIR}/plots/"
    
    # Download model outputs
    echo "Downloading model outputs..."
    scp -r "${REMOTE_USER}@${REMOTE_HOST}:${REMOTE_BASE_DIR}/models/${EXP_NAME}*" "${LOCAL_EXP_DIR}/models/"
    
    # Download specific log files
    echo "Downloading experiment logs..."
    find_and_download_logs "${EXP_NAME}" "${LOCAL_EXP_DIR}" "${REMOTE_HOST}"
    
    echo "Results downloaded to: ${LOCAL_EXP_DIR}"
}

# Function to find and download specific log files
find_and_download_logs() {
    local EXP_NAME=$1
    local LOCAL_DIR=$2
    local REMOTE_HOST=$3
    
    # Download training logs
    scp "${REMOTE_USER}@${REMOTE_HOST}:${REMOTE_BASE_DIR}/logs/run_*/training_log.txt" "${LOCAL_DIR}/logs/"
    
    # Download inference samples
    scp "${REMOTE_USER}@${REMOTE_HOST}:${REMOTE_BASE_DIR}/logs/run_*/inference_samples_epoch_*.txt" "${LOCAL_DIR}/logs/"
}

# Main execution
main() {
    if [ $# -ne 2 ]; then
        echo "Usage: $0 <experiment_name> <remote_host>"
        echo "Example: $0 name_number_query_sft your.server.address"
        exit 1
    fi
    
    local EXP_NAME=$1
    local REMOTE_HOST=$2
    
    # Check if experiment exists on remote
    if ssh "${REMOTE_USER}@${REMOTE_HOST}" "test -d ${REMOTE_BASE_DIR}/logs/run_*"; then
        download_experiment_results "${EXP_NAME}" "${REMOTE_HOST}"
    else
        echo "Error: No results found for experiment ${EXP_NAME} on ${REMOTE_HOST}"
        exit 1
    fi
}

# Execute main function with provided arguments
main "$@"
