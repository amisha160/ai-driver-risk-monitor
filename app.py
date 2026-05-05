from flask import Flask, render_template, request, jsonify, send_file
import numpy as np
import pickle
from PIL import Image
import pandas as pd
from datetime import datetime
import os
import io

app = Flask(__name__)

# =========================
# LOAD MODEL
# Make sure model.pkl exists — generate it from driver_behavior_final.py
# =========================
model = pickle.load(open("model.pkl", "rb"))

HISTORY_FILE = "history.csv"

# Create history CSV file if it doesn't exist
if not os.path.exists(HISTORY_FILE):
    pd.DataFrame(columns=[
        "time",
        "speed_variation",
        "acceleration",
        "steering",
        "driver_condition",
        "image_score",
        "risk_percent",
        "status",
        "suggestion"
    ]).to_csv(HISTORY_FILE, index=False)


# =========================
# FEATURE FUNCTIONS
# =========================

def text_feature(text):
    """Extract risk score from driver condition text input."""
    keywords = ["drowsy", "sleepy", "phone", "danger", "tired", "distracted", "aggressive"]
    count = sum(1 for k in keywords if k in text.lower())
    return count / len(keywords)


def analyze_image(img_file):
    """
    Analyze uploaded driver image.
    Returns a normalized brightness score and a state description.
    """
    if img_file and img_file.filename != "":
        img = Image.open(img_file).convert("RGB")
        arr = np.array(img) / 255.0
        score = float(np.mean(arr))
        if score < 0.30:
            state = "Low visibility / possible drowsy face detected"
        elif score < 0.60:
            state = "Normal alertness detected"
        else:
            state = "Bright / alert driver detected"
        return score, state
    return 0.5, "No image uploaded"


# =========================
# DRIVER STATE CLASSIFIER
# FIXED: Matches patent — Safe | Distracted | Drowsy | Aggressive
# =========================

def classify(risk):
    """
    Classify driver state based on risk score (0-1).
    Returns: status label, color hex, suggestion message.
    """
    if risk < 0.30:
        return (
            "SAFE",
            "#00ff88",
            "Driving pattern is stable. Keep it up!"
        )
    elif risk < 0.50:
        return (
            "DISTRACTED",
            "#ffd000",
            "You seem distracted. Please focus on the road."
        )
    elif risk < 0.75:
        return (
            "DROWSY",
            "#ff8c00",
            "Drowsiness detected. Pull over and take a break."
        )
    else:
        return (
            "AGGRESSIVE",
            "#ff3b3b",
            "Aggressive driving detected! Slow down immediately."
        )


# =========================
# SAVE TO HISTORY
# =========================

def save_history(s1, s2, s3, text, img_score, risk, status, suggestion):
    """Save each prediction session to history CSV."""
    row = {
        "time":           datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "speed_variation": s1,
        "acceleration":    s2,
        "steering":        s3,
        "driver_condition": text,
        "image_score":     round(img_score, 3),
        "risk_percent":    round(risk * 100, 2),
        "status":          status,
        "suggestion":      suggestion
    }
    df = pd.read_csv(HISTORY_FILE)
    df = pd.concat([df, pd.DataFrame([row])], ignore_index=True)
    df.to_csv(HISTORY_FILE, index=False)


# =========================
# ROUTES
# =========================

@app.route("/")
def home():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():
    try:
        # Get sensor inputs from form
        s1 = float(request.form["s1"])   # Speed variation
        s2 = float(request.form["s2"])   # Acceleration
        s3 = float(request.form["s3"])   # Steering stability

        # Text behavioral input
        text     = request.form.get("text", "")
        txt_feat = text_feature(text)

        # Image input
        img_file            = request.files.get("image")
        img_score, img_state = analyze_image(img_file)

        # Build unified multimodal feature vector (6 features)
        # [speed, acceleration, steering, image_score, video_proxy, text_score]
        # Note: img_score is used as video proxy since no live video stream
        features = np.array([[s1, s2, s3, img_score, img_score, txt_feat]])

        # Predict risk probability using Logistic Regression model
        risk = float(model.predict_proba(features)[0][1])

        # Classify into one of the 4 driver states
        status, color, suggestion = classify(risk)

        # Save to history
        save_history(s1, s2, s3, text, img_score, risk, status, suggestion)

        return jsonify({
            "risk":        round(risk * 100, 2),
            "status":      status,
            "color":       color,
            "suggestion":  suggestion,
            "image_state": img_state
        })

    except Exception as e:
        return jsonify({"error": str(e)})


@app.route("/history")
def history():
    """Return last 10 driving sessions as JSON."""
    df = pd.read_csv(HISTORY_FILE)
    return df.tail(10).to_json(orient="records")


@app.route("/download_report")
def download_report():
    """Download full driving history as CSV report."""
    df = pd.read_csv(HISTORY_FILE)
    buffer = io.StringIO()
    df.to_csv(buffer, index=False)
    buffer.seek(0)
    return send_file(
        io.BytesIO(buffer.getvalue().encode()),
        mimetype="text/csv",
        as_attachment=True,
        download_name="driver_report.csv"
    )


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
