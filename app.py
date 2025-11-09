# app.py
from flask import Flask, request, jsonify, send_from_directory
from search import load_resources, perform_search
from flask_cors import CORS
import os

app = Flask(__name__)
CORS(app)  # Allow calls from index.html even if it's served elsewhere

# Load your embeddings/model only once when server starts
load_resources()

# RAG/app.py (Fixed search_endpoint)

@app.route("/api/search", methods=["POST"])
def search_endpoint():
    print("--- API Search endpoint successfully HIT! ---")
    
    # 1. Correctly get the JSON data from the request body
    try:
        data = request.get_json()
        query = data.get("query", "")
    except Exception:
        # Handle cases where the request body isn't valid JSON
        return jsonify({"error": "Invalid JSON payload"}), 400
        
    # 2. Check the query
    if not query.strip():
        return jsonify({"error": "Empty query"}), 400
    
    # 3. Perform search
    results = perform_search(query)

    # 4. Format results (rest of the logic is correct)
    formatted = []
    for r in results:
        raw_path = r.get("File Path", "")
        # Use os.path.basename to get just the filename (e.g., "boho1.jpg")
        file_name = os.path.basename(raw_path) if raw_path else None
        
        # This converts the local file name into a Flask-served URL: /images/boho1.jpg
        file_url = f"/images/{file_name}" if file_name else None

        formatted.append({
            "rank": r.get("rank"),
            "title": r.get("Generated Title"),
            "style": r.get("Style"),
            "file_url": file_url, # Renamed to file_url for clarity
            "score": r.get("score"),
        })

    return jsonify(formatted)

@app.route('/')
def index():
    """Serve the HTML frontend file."""
    # This route is correct, assuming index2.html is in the same directory as app.py
    return send_from_directory('.', 'index2.html')


# Serve images from the 'static' folder
@app.route("/images/<path:filename>")
def serve_image(filename):
    return send_from_directory("static", filename)


if __name__ == "__main__":
    app.run(debug=True)
