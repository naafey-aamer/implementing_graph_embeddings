#!/bin/bash

# Check if the number of edges parameter is provided
if [ -z "$1" ]
then
  echo "Please provide the number of edges to consider."
  echo "Usage: ./setup_data.sh <number_of_edges>"
  exit 1
fi

NUM_EDGES=$1

# Step 1: Download the data using Selenium
echo "Downloading data using Selenium..."
python3 download_protein_data.py

# Step 2: Check if the file was downloaded and moved successfully
if [ ! -f "9606.protein.links.v12.0.txt.gz" ]; then
    echo "File download failed or file not found."
    exit 1
fi

# Step 3: Unzip the file
echo "Unzipping the downloaded file..."
gunzip 9606.protein.links.v12.0.txt.gz

# Step 4: Run the Python script to process the data
echo "Processing the data..."
python3 process_data.py $NUM_EDGES

# Notify the user
echo "Data downloaded, processed, and saved successfully."
