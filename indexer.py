#!/usr/bin/env python3

"""
This script loads 'database.json', corrects the file paths to be absolute
Windows paths based on the script's current location, saves the corrected
data, and then generates sentence embeddings for the 'Generated Description'
field, saving them to 'embeddings.npy'.
"""

import json
import numpy as np
from sentence_transformers import SentenceTransformer
import sys
import os # <-- We need this for path manipulation

def main():
    #Load Model
    model_name = 'all-mpnet-base-v2'
    print(f"Loading sentence-transformer model: {model_name}...")
    try:
        model = SentenceTransformer(model_name)
    except Exception as e:
        print(f"Error loading model: {e}")
        print("Please ensure 'sentence-transformers' is installed: pip install sentence-transformers")
        sys.exit(1)
    print("Model loaded successfully.")

    #Load Data from JSON file
    data_filename = 'database.json'
    print(f"Loading data from {data_filename}...")
    try:
        with open(data_filename, 'r', encoding='utf-8') as f:
            data = json.load(f)
    except FileNotFoundError:
        print(f"Error: {data_filename} not found.")
        print(f"Please make sure '{data_filename}' is in the same directory.")
        sys.exit(1)
    except json.JSONDecodeError:
        print(f"Error: Could not decode {data_filename}. Check if it's valid JSON.")
        sys.exit(1)
    
    if not isinstance(data, list):
        print(f"Error: Expected {data_filename} to contain a JSON list of objects.")
        sys.exit(1)

    print(f"Loaded {len(data)} items.")

    # SET THE FILE PATHS
    #print("Correcting file paths...")
    
    
    script_dir = os.path.abspath(os.path.dirname(__file__))
    
    items_fixed = 0
    items_skipped = 0

    for item in data:
        old_path = item.get('File Path') # Use .get() for safety
        
        if not old_path or 'bathroom' not in old_path:
            print(f"Warning: 'File Path' is missing or invalid for item: {item.get('File Name')}. Skipping path correction.")
            items_skipped += 1
            continue

        try:
            # Find the start of the relevant part of the path
            rel_path_start_index = old_path.index('bathroom')
            
            # Get the path from "bathroom" onwards: "bathroom/minimalist/..."
            rel_path_linux = old_path[rel_path_start_index:]
            
            # Convert the Linux-style path '/' to the local OS's separator '\'
            rel_path_os = rel_path_linux.replace('/', os.sep)
            
            # Create the new, full, absolute path
            # e.g., "C:\Users\Khushi\...\CornHacks" + "bathroom\minimalist\..."
            new_path = os.path.join(script_dir, rel_path_os)
            
            # Update the path in our data
            item['File Path'] = new_path
            items_fixed += 1

        except Exception as e:
            print(f"Error processing path for item: {item}. Error: {e}")
            items_skipped += 1

    print(f"Corrected paths for {items_fixed} items. Skipped {items_skipped} items.")

    # SAVE THE JSON
    try:
        with open(data_filename, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=4)
        print(f"Successfully saved corrected paths back to {data_filename}.")
    except Exception as e:
        print(f"Error saving updated {data_filename}: {e}")
        sys.exit(1)

    # Extract Descriptions
    print("Extracting descriptions for encoding...")
    try:
        descriptions = [
            item.get('Generated Description') if item.get('Generated Description') is not None else "" 
            for item in data
        ]
    except Exception as e:
        print(f"Error extracting descriptions: {e}")
        sys.exit(1)

    # Encode Descriptions
    print(f"Encoding {len(descriptions)} descriptions... (This may take a moment)")
    try:
        embeddings = model.encode(descriptions, show_progress_bar=True)
        print("Encoding complete.")
    except Exception as e:
        print(f"An error occurred during encoding: {e}")
        sys.exit(1)

    # Save Embeddings
    output_filename = 'embeddings.npy'
    try:
        np.save(output_filename, embeddings)
        print(f"\nSuccessfully saved embeddings to {output_filename}")
        print(f"Shape of saved array: {embeddings.shape}")
    except Exception as e:
        print(f"Error saving embeddings to {output_filename}: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()