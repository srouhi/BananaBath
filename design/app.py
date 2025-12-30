import os
import sys
from flask import Flask, request, jsonify, send_from_directory, abort
from flask_cors import CORS

# --- 1. Import Business Logic ---
try:
    # We import the entire module to access its functions
    import search as search_engine
except ImportError:
    print("Error: search.py not found.")
    print("Please make sure app.py and search.py are in the same directory.")
    sys.exit(1)

# --- 2. Initialization ---
app = Flask(__name__)
CORS(app)  # Allow cross-origin requests

# Define absolute paths for reliability
# APP_ROOT is the directory where this script (app.py) lives
APP_ROOT = os.path.dirname(os.path.abspath(__file__))
# STATIC_DIR is the 'static' folder inside the APP_ROOT
STATIC_DIR = os.path.join(APP_ROOT, 'static')

# --- 3. Load Resources Once on Start ---
print("Loading resources...")
try:
    # Call the load function from your search module
    search_engine.load_resources()
    print("Resources loaded successfully.")
except Exception as e:
    print(f"FATAL ERROR: Could not load resources: {e}")
    sys.exit(1)

# --- 4. API Route ---
@app.route("/api/search", methods=["POST"])
def search_endpoint():
    """
    Handles search requests from the frontend.
    Expects JSON: {"query": "some text"}
    Returns JSON: [{"file_url": "...", "title": "...", ...}]
    """
    print("--- API Search endpoint hit ---")
    
    # 1. Get and validate JSON payload
    try:
        data = request.get_json()
        query = data.get("query", "")
    except Exception as e:
        print(f"Error parsing request JSON: {e}")
        return jsonify({"error": "Invalid JSON payload"}), 400
        
    # 2. Validate the query itself
    if not query.strip():
        print("Empty query received.")
        return jsonify({"error": "Empty query"}), 400
    
    # 3. Perform search
    try:
        # Delegate all search and result-formatting logic to search.py
        # We assume perform_search returns a list of dicts
        # ready to be turned into JSON.
        results = search_engine.perform_search(query, top_k=12)
        
        # 4. Return results
        return jsonify(results)
        
    except Exception as e:
        print(f"Error during search: {e}")
        return jsonify({"error": "An internal error occurred during search."}), 500

# --- 5. Static File Serving ---
@app.route('/static/<path:filepath>')
def serve_static_file(filepath):
    """
    Serves any file from the 'static' directory.
    The frontend will request URLs like: /static/boho/image1.jpg
    """
    try:
        # Safely serves files from your 'static' directory
        return send_from_directory(STATIC_DIR, filepath)
    except FileNotFoundError:
        print(f"File not found in static: {filepath}")
        abort(404)

# --- 6. Root/Test Route ---
@app.route('/')
def index():
    """
    Serves a basic index.html file from the root directory for testing.
    """
    index_path = os.path.join(APP_ROOT, 'index.html')
    if os.path.exists(index_path):
        return send_from_directory(APP_ROOT, 'index.html')
    
    # Provide a fallback message if index.html is missing
    return "<h1>API is running!</h1><p>Send POST requests to /api/search</p>"

# --- 7. Run the App ---
if __name__ == "__main__":
    # debug=True reloads the server on code changes
    # host='0.0.0.0' makes it accessible on your network (optional)
    app.run(debug=True, port=5000)