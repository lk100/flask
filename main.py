from flask import Flask, request, jsonify
from flask_cors import CORS
from transformers import pipeline
import os

# Load the Hugging Face model
classifier = pipeline(
    "sentiment-analysis",
    model="lk1307/love_model",
    token="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",
    framework="pt"  # Force PyTorch
)

# Initialize the Flask app
app = Flask(__name__)

# Enable CORS for the frontend domain
CORS(app, origins=["https://mindbliss.up.railway.app"])

# Define a route for sentiment analysis
@app.route("/predict", methods=["POST"])
def submit_journal():
    data = request.get_json()
    user_text = data.get("text")
    user_id = data.get("user_id")

    if not user_text or not user_id:
        return jsonify({"error": "Missing text or user_id"}), 400

    try:
        print("Received text:", user_text)  # Log the received text
        result = classifier(user_text)
        emotion = result[0]["label"]
        print("Predicted emotion:", emotion)  # Log the predicted emotion
        return jsonify({"success": True, "emotion": emotion})
    except Exception as e:
        print(f"Error during prediction: {e}")  # Log the error for debugging
        return jsonify({"error": str(e)}), 500

# Run the Flask app
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))  # Default to 5000 if not set by Railway
    app.run(debug=True, host="0.0.0.0", port=port)
