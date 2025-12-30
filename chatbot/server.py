# server.py
from flask import Flask, request, jsonify
from flask_cors import CORS
import chat  # chat.py file

app = Flask(__name__)
CORS(app)  # allow frontend to call this API

@app.route("/chat", methods=["POST"])
def chat_endpoint():
    data = request.get_json()
    user_input = data.get("message", "")

    try:
        # existing generate() function
        reply = chat.generate_for_api(user_input)
        return jsonify({"reply": reply})
    except Exception as e:
        print("Error in /chat:", e)
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    app.run(debug=True)
