from flask import Flask, request, jsonify, send_from_directory
from search import load_resources, perform_search
from flask_cors import CORS
import os
import chat  # chat.py file

# 1. Initialization and Resource Loading
app = Flask(__name__)
CORS(app) # Initializes CORS once
load_resources() # Load search resources when the app starts


# 2. Chat Endpoint
@app.route("/chat", methods=["POST"])
def chat_endpoint():
    data = request.get_json()
    user_input = data.get("message", "")

    try:
        # Note: chat.generate_for_api is assumed to handle the chat interaction 
        # using the Gemini API or other logic defined in chat.py
        reply = chat.generate_for_api(user_input)
        return jsonify({"reply": reply})
    except Exception as e:
        print("Error in /chat:", e)
        return jsonify({"error": str(e)}), 500


# 3. Search Endpoint
@app.route("/api/search", methods=["POST"])
def search_endpoint():
    print("--- API Search endpoint successfully HIT! ---")
    
    try:
        data = request.get_json()
        query = data.get("query", "")
    except Exception:
        return jsonify({"error": "Invalid JSON payload"}), 400
        
    if not query.strip():
        return jsonify({"error": "Empty query"}), 400
    
    results = perform_search(query)

    formatted = []
    for r in results:
        # Get style (lowercase for folder name) and file name
        style = r.get("Style", "").lower()
        file_name = r.get("File Name")
        
        # CRITICAL FIX: The URL structure is corrected to match the physical path: 
        # static/{style}/{file_name}
        if file_name and style:
            file_url = f"/static/{style}/{file_name}" 
        else:
            file_url = None

        formatted.append({
            "rank": r.get("rank"),
            "title": r.get("Generated Title"),
            "style": r.get("Style"),
            "file_url": file_url,
            "score": r.get("score"),
        })

    return jsonify(formatted)


# 4. Frontend and Static File Serving
@app.route('/')
def index():
    """Serve the HTML frontend file from the root directory."""
    # Assumes index2.html is in the same directory as app.py
    return send_from_directory('.', 'index2.html')


# NOTE ON STATIC FILES:
# The problematic custom route was REMOVED.
# Flask's built-in static file handler automatically handles requests for 
# files under the /static/ prefix, making this custom route unnecessary 
# and preventing the 404 error you experienced.

if __name__ == "__main__":
    app.run(debug=True)