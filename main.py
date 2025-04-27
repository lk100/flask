from flask import Flask, request, jsonify
from flask_cors import CORS  # Import CORS
from transformers import pipeline

# Load the Hugging Face model
classifier = pipeline(
    "sentiment-analysis",
    model="lk1307/love_model",
    token="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",
    framework="pt"  # Force PyTorch
)

# Initialize the Flask app
app = Flask(__name__)

# Enable CORS for the frontend domain (adjust as needed)
CORS(app, origins=["https://mindbliss.up.railway.app"])

# Define a route for sentiment analysis
@app.route("/predict", methods=["POST"])
def submit_journal():
    data = request.get_json()
    user_text = data.get("text")
    user_id = data.get("user_id")

    if not user_text or not user_id:
        return jsonify({"error": "Missing text or user_id"}), 400

    # Get sentiment analysis result
    result = classifier(user_text)
    emotion = result[0]["label"]

    # Simulate saving to the database (you can connect to a real database here)
    return jsonify({"success": True, "emotion": emotion})

# Run the Flask app
if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5000)
