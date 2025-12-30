#!/usr/bin/env python3
"""
This script runs a command-line interface (CLI) to test the advanced RAG
search functionality. It loads the database, embeddings, and models once,
then accepts user queries in a loop.

This code uses a simple, rule-based
parser to handle "A and not B" queries.

It uses a 3-stage RAG pipeline:
1. PARSE: A robust local function parses the user's query into
   a "pure" positive part and a "negative" part.
2. RETRIEVE: The sentence-transformer model gets embeddings for BOTH parts.
3. RE-RANK: It ranks all images based on (Positive Similarity - Negative Similarity)
   to understand "A and not B" queries.
"""

import json
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import sys
import os
import re 
import webbrowser
from PIL import Image

# --- 1. Global Variables ---
model = None
data = None
embeddings = None

def load_resources():
    """
    Loads all required models and data files into global variables.
    This is run once at startup.
    """
    global model, data, embeddings
    
    print("Loading resources...")
    try:
        print("Loading sentence transformer model: all-mpnet-base-v2...")
        model = SentenceTransformer('all-mpnet-base-v2')
        
        print("Loading database.json...")
        with open('database.json', 'r', encoding='utf-8') as f:
            data = json.load(f)
            
        print("Loading embeddings.npy...")
        # Force-load as a numpy array
        embeddings = np.load('embeddings.npy')
        if not isinstance(embeddings, np.ndarray):
             embeddings = embeddings.numpy() # Convert from tensor if it was saved as one
        
        # Check for data/embedding count mismatch
        data_len = len(data)
        embed_len = embeddings.shape[0]
        
        # Filter data if it contains items (like logs) that embeddings don't
        if data_len > embed_len:
            print(f"Warning: Data/Embedding mismatch! JSON has {data_len} items, Embeddings has {embed_len} rows.")
            # We assume embeddings were generated only for items with 'Generated Description'
            # Filter the data list to match the embeddings
            data = [item for item in data if item.get('Generated Description') is not None]
            print(f"Data list filtered to {len(data)} items to match embeddings.")
        
        if len(data) != embeddings.shape[0]:
             print(f"CRITICAL WARNING: Data ({len(data)}) and Embedding ({embeddings.shape[0]}) counts still do not match. Search results may be incorrect.")

        
    except FileNotFoundError as e:
        print(f"Error: Missing file! {e.filename}")
        print("Please make sure 'database.json' and 'embeddings.npy' are in this directory.")
        sys.exit(1)
    except Exception as e:
        print(f"An error occurred during loading: {e}")
        print("Please ensure libraries are installed: pip install numpy scikit-learn sentence-transformers pillow")
        sys.exit(1)
        
    print(f"\nSuccessfully loaded {len(data)} items and {embeddings.shape[0]} embeddings.")
    print("--- RAG Search is Ready ---")

# --- v5: CORRECTED PARSER ---
def get_structured_query(user_query):
    """
    (OFFLINE VERSION - v5)
    Parses a natural language query into positive and negative terms
    using a more robust regex split.
    """
    print(f"Parsing query locally: \"{user_query}\"...")
    
    # Define negative keywords. We look for them surrounded by spaces
    # to avoid matching "without" inside a word, for example.
    negative_triggers = [
        "but not", "without", "and not", "except", 
        "do not have", "don't have", "not including", "excluding"
    ]
    
    # Create a regex pattern: ( but not | without | ...)
    # The ( ) capture group is key: re.split will include the delimiter in the result
    trigger_pattern = re.compile(
        r' ' + r'|'.join(re.escape(t) for t in negative_triggers) + r' ', 
        re.IGNORECASE
    )
    
    match = trigger_pattern.search(user_query)
    
    positive_query = user_query
    negative_query = ""
    
    if match:
        # If a trigger is found:
        # split_index is the *start* of the trigger (e.g., the space before "do not have")
        split_index = match.start() 
        # end_index is the *end* of the trigger (e.g., the space after "do not have")
        end_index = match.end()
        
        # The positive part is everything *before* the trigger
        positive_query = user_query[:split_index].strip()
        
        # The negative part is everything *after* the trigger
        negative_query = user_query[end_index:].strip()
            
    parsed_json = {
        "positive_query": positive_query,
        "negative_query": negative_query
    }
    
    # This parse should now be correct:
    # {'positive_query': 'return minimalistic styles', 'negative_query': 'white tones'}
    print(f"Local Parser Result: {parsed_json}")
    return parsed_json


def perform_search(user_query):
    """
    Handles the 3-stage (Parse-Retrieve-Re-rank) search logic.
    """
    global model, data, embeddings
    
    try:
        # 1. PARSE: Use the local parser to understand positive/negative intent
        parsed_query = get_structured_query(user_query)
        
        # 2. RETRIEVE: Get embedding for the CLEAN positive query
        positive_query_text = parsed_query.get('positive_query', user_query)
        positive_embedding = model.encode(positive_query_text)
        # Ensure it's a 2D numpy array
        if not isinstance(positive_embedding, np.ndarray):
            positive_embedding = positive_embedding.cpu().numpy()
        positive_embedding = positive_embedding.reshape(1, -1) 

        # Calculate positive similarities
        similarities_positive = cosine_similarity(positive_embedding, embeddings)[0]
        
        negative_query = parsed_query.get('negative_query')
        
        if negative_query:
            # If a negative part was extracted, create a "penalty" embedding
            print(f"Creating penalty vector for: \"{negative_query}\"")
            negative_embedding = model.encode(negative_query)
            # Ensure it's a 2D numpy array
            if not isinstance(negative_embedding, np.ndarray):
                negative_embedding = negative_embedding.cpu().numpy()
            negative_embedding = negative_embedding.reshape(1, -1)

            similarities_negative = cosine_similarity(negative_embedding, embeddings)[0]
            
            # 3. RE-RANK: Subtract negative scores from positive scores
            penalty_weight = 1.0 
            final_scores = similarities_positive - (similarities_negative * penalty_weight)
            print("Using Positive-Negative Re-ranking...")
        else:
            # If no negative query, just use the positive scores
            final_scores = similarities_positive
            print("Using Simple Semantic Search...")
        
        # --- (Section 4 & 5: Filter for top 6 with score < 0.5) ---
        
        # 4. Get ALL indices, sorted from highest score to lowest
        all_sorted_indices = np.argsort(final_scores)[::-1]
        
        # 5. Format the results:
        results = []
        rank_counter = 1
        for idx in all_sorted_indices:
            # Stop if we already have 6 results
            if len(results) >= 6:
                break
                
            score = float(final_scores[idx])
            
            # Check the new condition: score must be less than 0.5
            if score < 0.5:
                if idx < len(data):
                    item_data = data[idx]
                    results.append({
                        **item_data, 
                        'score': score,
                        'rank': rank_counter
                    })
                    rank_counter += 1
                else:
                    print(f"Skipping index {idx}, out of bounds for data (len {len(data)})")

        return results

    except Exception as e:
        print(f"Error during search: {e}")
        return []

def show_image_in_viewer(file_path):
    """
    Opens an image file in the default OS image viewer.
    """
    if not file_path:
        print("Error: No file path provided for top result.")
        return
        
    if not os.path.exists(file_path):
        print(f"\n--- Error opening image ---")
        print(f"Could not find file: {file_path}")
        print("Please check that 'database.json' paths are correct.")
        print("You may need to re-run 'create_embeddings.py' to fix paths.")
        return

    try:
        if Image:
            print(f"\nOpening top result in image viewer: {file_path}")
            img = Image.open(file_path)
            img.show()
        else:
            print(f"\n(Pillow not found) Opening top result in browser: {file_path}")
            webbrowser.open(f'file://{os.path.abspath(file_path)}')
    except Exception as e:
        print(f"Error opening image: {e}")
        try:
            webbrowser.open(f'file://{os.path.abspath(file_path)}')
        except Exception as e2:
            print(f"Webbrowser fallback failed: {e2}")

def main_cli_loop():
    """
    Runs the main interactive command-line loop.
    """
    # Load all resources once at the start
    load_resources()

    while True:
        print("\n" + "="*50)
        query = input("Enter your search query (or 'q' to quit): ")
        
        if query.lower() == 'q':
            print("Exiting. Goodbye!")
            break
        if not query.strip():
            continue
            
        print("Searching...")
        results = perform_search(query)
        
        print("\n--- Top Search Results ---")
        if not results:
            print("No results found.")
            continue
            
        for item in results:
            file_name = item.get('File Name', 'N/A')
            if file_name == 'N/A' and item.get('File Path'):
                file_name = os.path.basename(item['File Path'])

            print(f"\nRank {item['rank']} (Score: {item['score']:.4f})")
            print(f"  Title: {item.get('Generated Title', 'N/A')}")
            print(f"  Style: {item.get('Style', 'N/A')}")
            print(f"  File:  {file_name}")
            
        if results:
            top_result_path = results[0].get('File Path')
            show_image_in_viewer(top_result_path)

if __name__ == "__main__":
    main_cli_loop()