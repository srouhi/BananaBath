#!/usr/bin/env python3
"""
This script is the core search engine.
It can be imported by a web server (like app.py) or
run directly in the terminal for testing.

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
try:
    from PIL import Image
except ImportError:
    print("Warning: Pillow library not found. Run: pip install pillow")
    Image = None

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
        embeddings = np.load('embeddings.npy')
        
        # This check should now pass after running indexer.py
        data_len = len(data)
        embed_len = embeddings.shape[0]
        
        if data_len != embed_len:
             print(f"CRITICAL WARNING: Data ({data_len}) and Embedding ({embed_len}) counts do not match. Please re-run indexer.py!")
             sys.exit(1)
        
    except FileNotFoundError as e:
        print(f"Error: Missing file! {e.filename}")
        print("Please make sure 'database.json' and 'embeddings.npy' are in this directory.")
        print("You may need to run 'python indexer.py' first.")
        sys.exit(1)
    except Exception as e:
        print(f"An error occurred during loading: {e}")
        print("Please ensure libraries are installed: pip install numpy scikit-learn sentence-transformers pillow")
        sys.exit(1)
        
    print(f"\nSuccessfully loaded {len(data)} items and {embeddings.shape[0]} embeddings.")
    print("--- RAG Search is Ready ---")

def get_structured_query(user_query):
    """
    (OFFLINE VERSION - v5)
    Parses a natural language query into positive and negative terms
    using a more robust regex split.
    """
    print(f"Parsing query locally: \"{user_query}\"...")
    
    # Define negative keywords.
    negative_triggers = [
        "but not", "without", "and not", "except", 
        "do not have", "don't have", "not including", "excluding"
    ]
    
    # Create a case-insensitive regex pattern to split the query
    trigger_pattern = re.compile(
        r' ' + r'|'.join(re.escape(t) for t in negative_triggers) + r' ', 
        re.IGNORECASE
    )
    
    match = trigger_pattern.search(user_query)
    
    positive_query = user_query
    negative_query = ""
    
    if match:
        # If a trigger is found:
        split_index = match.start() 
        end_index = match.end()
        
        # The positive part is everything *before* the trigger
        positive_query = user_query[:split_index].strip()
        
        # The negative part is everything *after* the trigger
        negative_query = user_query[end_index:].strip()
            
    parsed_json = {
        "positive_query": positive_query,
        "negative_query": negative_query
    }
    
    print(f"Local Parser Result: {parsed_json}")
    return parsed_json


def perform_search(user_query, top_k=12):
    """
    Handles the 3-stage (Parse-Retrieve-Re-rank) search logic.
    Returns a list of the top_k result dictionaries.
    """
    global model, data, embeddings
    
    try:
        # 1. PARSE: Use the local parser to understand positive/negative intent
        parsed_query = get_structured_query(user_query)
        
        # 2. RETRIEVE: Get embedding for the CLEAN positive query
        positive_query_text = parsed_query.get('positive_query', user_query)
        # Force conversion to numpy array
        positive_embedding = model.encode(positive_query_text, convert_to_numpy=True).reshape(1, -1) 

        # Calculate positive similarities
        similarities_positive = cosine_similarity(positive_embedding, embeddings)[0]
        
        negative_query = parsed_query.get('negative_query')
        
        if negative_query:
            # If a negative part was extracted, create a "penalty" embedding
            print(f"Creating penalty vector for: \"{negative_query}\"")
            # Force conversion to numpy array
            negative_embedding = model.encode(negative_query, convert_to_numpy=True).reshape(1, -1)

            similarities_negative = cosine_similarity(negative_embedding, embeddings)[0]
            
            # 3. RE-RANK: Subtract negative scores from positive scores
            penalty_weight = 1.0 
            final_scores = similarities_positive - (similarities_negative * penalty_weight)
            print("Using Positive-Negative Re-ranking...")
        else:
            # If no negative query, just use the positive scores
            final_scores = similarities_positive
            print("Using Simple Semantic Search...")
        
        # 4. Get Top K indices
        # Get the indices of the top K scores in descending order
        top_k_indices = np.argsort(final_scores)[-top_k:][::-1]
        
        # 5. Format the results
        results = []
        rank_counter = 1
        for idx in top_k_indices:
            score = float(final_scores[idx])
            
            item_data = data[idx]
            results.append({
                **item_data, 
                'score': score,
                'rank': rank_counter
            })
            rank_counter += 1

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
        print("You may need to re-run 'indexer.py' to fix paths.")
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


def main_cli_loop():
    """
    Runs the main interactive command-line loop for testing.
    """
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
        # Get 6 results, all with score < 0.5 (as you requested)
        all_results = perform_search(query, top_k=len(data)) # Get all results
        
        # Apply your custom filtering
        filtered_results = [r for r in all_results if r['score'] < 0.5][:6]
        
        print("\n--- Top Search Results (Score < 0.5, Max 6) ---")
        if not filtered_results:
            print("No results found matching your criteria.")
            continue
            
        for item in filtered_results:
            file_name = item.get('File Name', 'N/A')
            print(f"\nRank {item['rank']} (Score: {item['score']:.4f})")
            print(f"  Title: {item.get('Generated Title', 'N/A')}")
            print(f"  Style: {item.get('Style', 'N/A')}")
            print(f"  File:  {file_name}")
            
        if filtered_results:
            top_result_path = filtered_results[0].get('File Path')
            show_image_in_viewer(top_result_path)

# This check ensures this code only runs if you execute `python search.py`
# It will NOT run when `app.py` imports it, which is correct.
if __name__ == "__main__":
    print("This file is a library. Running CLI test mode...")
    main_cli_loop()