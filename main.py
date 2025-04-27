from flask import Flask, request, jsonify
from flask_cors import CORS
from transformers import pipeline
import os
import logging

# Set up logging
logging.basicConfig(level=logging.DEBUG)

# Initialize the Flask app
app = Flask(__name__)

# Enable CORS for the frontend domain
CORS(app, origins=["https://mindbliss.up.railway.app"])

# Initialize the model globally
classifier = None

# Function to load the model at app startup
def load_model():
    global classifier
    try:
        logging.info("Loading model...")
        classifier = pipeline(
            "sentiment-analysis",
            model="lk1307/love_model",
            token="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",
            framework="pt"  # Force PyTorch
        )
        logging.info("Model loaded successfully.")
    except Exception as e:
        logging.error(f"Error loading model: {e}")

# Load the model when the app starts
load_model()

# Define a route for sentiment analysis
@app.route("/predict", methods=["POST"])
def submit_journal():
    # Get input data
    data = request.get_json()
    user_text = data.get("text")
    user_id = data.get("user_id")

    if not user_text or not user_id:
        return jsonify({"error": "Missing text or user_id"}), 400

    try:
        logging.debug(f"Received text: {user_text}")  # Log the received text
        result = classifier(user_text)
        emotion = result[0]["label"]
        logging.debug(f"Predicted emotion: {emotion}")  # Log the predicted emotion
        return jsonify({"success": True, "emotion": emotion})
    except Exception as e:
        logging.error(f"Error during prediction: {e}")  # Log the error for debugging
        return jsonify({"error": str(e)}), 500

# Run the Flask app
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))  # Default to 5000 if not set by Railway
    app.run(debug=True, host="0.0.0.0", port=port)
