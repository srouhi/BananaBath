#!/usr/bin/env python3

"""
This script reads 'CornhacksData.xlsx', CLEANS it, corrects all file paths
to your local 'static' subfolder structure, creates 'database.json',
and then generates a new, clean 'embeddings.npy' file.

RUN THIS SCRIPT ONCE.
"""

import json
import numpy as np
from sentence_transformers import SentenceTransformer
import sys
import os
import pandas as pd

def main():
    # --- 1. Load Model ---
    model_name = 'all-mpnet-base-v2'
    print(f"Loading sentence-transformer model: {model_name}...")
    try:
        model = SentenceTransformer(model_name)
    except Exception as e:
        print(f"Error loading model: {e}")
        print("Please ensure 'sentence-transformers' is installed: pip install sentence-transformers")
        sys.exit(1)
    print("Model loaded successfully.")

    # --- 2. Load Data from EXCEL file ---
    excel_filename = 'CornhacksData.xlsx'
    data_filename = 'database.json'
    output_filename = 'embeddings.npy' # Defined here for use in multiple sections
    
    print(f"Loading data from {excel_filename}...")
    try:
        df = pd.read_excel(excel_filename, sheet_name="Sheet1")
    except FileNotFoundError:
        print(f"Error: {excel_filename} not found in this directory.")
        print(f"Please make sure '{excel_filename}' is in the same directory as this script.")
        sys.exit(1)
    except Exception as e:
        print(f"Error reading Excel file. Make sure 'pandas' and 'openpyxl' are installed.")
        print(f"Error: {e}")
        sys.exit(1)
    
    print(f"Loaded {len(df)} total items from Excel.")

    # --- 3. CLEAN Data and FIX File Paths (Updated for 'image' folder) ---
    print("Cleaning data and correcting file paths...")
    
    # Get the absolute directory where this script is running (e.g., ...\design)
    script_dir = os.path.abspath(os.path.dirname(__file__))
    
    cleaned_data = []
    descriptions_to_embed = []
    
    # Convert DataFrame to list of dictionaries for processing
    data = df.to_dict('records')
    
    for item in data:
        item_style = item.get('Style')
        item_filename = item.get('File Name')
        item_desc = item.get('Generated Description')

        # Filter out invalid entries (e.g., NaN values)
        if not all([item_style, item_filename, item_desc, pd.notna(item_style), pd.notna(item_filename), pd.notna(item_desc)]):
            continue

        try:
            # 1. BUILD ABSOLUTE PATH: Includes the 'image' subdirectory
            # Example: C:\...\design\static\scandinavian\image\bathroom_scandinavian_68.jpg
            new_path = os.path.join(script_dir, 'static', item_style.lower(), 'images', item_filename)
            
            # 2. BUILD RELATIVE URL: This is what the web browser will request
            new_url = f"/static/{item_style.lower()}/images/{item_filename}"

            # 3. CRITICAL FILE EXISTENCE CHECK
            if not os.path.exists(new_path):
                print(f"WARNING: File not found at calculated path: {new_path}. Skipping item.")
                continue # Skip this item if the file is missing
            
            # Update the paths in our item
            item['File Path'] = new_path  # The path search.py uses to load the file
            item['file_url'] = new_url   # The URL the browser uses
            
            # Add this clean item to our new list
            cleaned_data.append(item)
            descriptions_to_embed.append(item_desc)

        except Exception as e:
            print(f"Error processing path for item: {item}. Error: {e}")

    print(f"Kept {len(cleaned_data)} valid image items. Removed {len(data) - len(cleaned_data)} invalid entries.")

    # --- 4. SAVE THE CLEANED JSON ---
    try:
        if not cleaned_data:
            print("WARNING: No valid items were kept. database.json will be an empty list.")
        
        # The 'w' mode ensures the file is OVERWRITTEN with the new data
        with open(data_filename, 'w', encoding='utf-8') as f: 
            json.dump(cleaned_data, f, indent=4) 
        print(f"Successfully saved cleaned data to {data_filename}.")
    except Exception as e:
        print(f"Error saving updated {data_filename}: {e}")
        sys.exit(1)

    # --- 5. Encode Descriptions ---
    if not descriptions_to_embed:
        print("Skipping encoding: No valid descriptions to process.")
        # If no items were processed, we still save an empty embeddings file
        np.save(output_filename, np.array([]))
        print(f"Saved empty array to {output_filename}.")
        return # Exit the main function

    print(f"Encoding {len(descriptions_to_embed)} descriptions... (This may take a moment)")
    try:
        embeddings = model.encode(
            descriptions_to_embed, 
            show_progress_bar=True, # Ensure the progress bar is shown
            convert_to_numpy=True
        )
        print("Encoding complete.")
    except Exception as e:
        print(f"An error occurred during encoding: {e}")
        sys.exit(1)

    # --- 6. Save Embeddings ---
    try:
        np.save(output_filename, embeddings)
        print(f"\nSuccessfully saved new embeddings to {output_filename}")
        print(f"Shape of saved array: {embeddings.shape}")
        if len(cleaned_data) != embeddings.shape[0]:
              print(f"CRITICAL WARNING: Final data count ({len(cleaned_data)}) and embedding count ({embeddings.shape[0]}) do not match.")

    except Exception as e:
        print(f"Error saving embeddings to {output_filename}: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()