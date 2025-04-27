from flask import Flask, request, jsonify
from flask_cors import CORS
from transformers import pipeline
import os
import logging

logging.basicConfig(level=logging.DEBUG)

app = Flask(__name__)
CORS(app, origins=["https://mindbliss.up.railway.app"])

# Load model once globally
logging.info("Loading model...")
classifier = pipeline(
    "sentiment-analysis",
    model="lk1307/love_model",
    token="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",
    framework="pt"
)
logging.info("Model loaded successfully.")

@app.route("/predict", methods=["POST"])
def submit_journal():
    data = request.get_json()
    user_text = data.get("text")
    user_id = data.get("user_id")

    if not user_text or not user_id:
        return jsonify({"error": "Missing text or user_id"}), 400

    try:
        logging.debug(f"Received text: {user_text}")
        result = classifier(user_text)
        emotion = result[0]["label"]
        logging.debug(f"Predicted emotion: {emotion}")
        return jsonify({"success": True, "emotion": emotion})
    except Exception as e:
        logging.error(f"Error during sentiment analysis: {e}")
        return jsonify({"error": "Failed to process the text."}), 500

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
