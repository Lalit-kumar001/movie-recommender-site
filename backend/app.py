from flask import Flask, request, jsonify
from flask_cors import CORS
from model import recommend   # ✅ correct


app = Flask(__name__)
CORS(app)  # allow requests from React frontend

@app.route("/recommend", methods=["POST"])
def recommend_movies():
    data = request.json
    movie = data.get("movie")
    
    if not movie:
        return jsonify({"error": "Movie name is required"}), 400

    try:
        results = recommend(movie)
        return jsonify(results)
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(debug=True)
