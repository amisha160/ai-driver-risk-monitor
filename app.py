from flask import Flask, render_template, request, jsonify, send_file
import numpy as np
import pickle
from PIL import Image
import pandas as pd
from datetime import datetime
import os
import io

app = Flask(__name__)

# Load model
model = pickle.load(open("model.pkl", "rb"))

HISTORY_FILE = "history.csv"

# Create history file automatically
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


def text_feature(text):
    keywords = ["drowsy", "sleepy", "phone", "danger", "tired"]
    count = sum(1 for k in keywords if k in text.lower())
    return count / len(keywords)


def analyze_image(img_file):
    if img_file and img_file.filename != "":
        img = Image.open(img_file).convert("RGB")
        arr = np.array(img) / 255.0
        score = float(np.mean(arr))

        if score < 0.30:
            state = "Low visibility / sleepy face"
        elif score < 0.60:
            state = "Normal alertness"
        else:
            state = "Bright / alert"

        return score, state

    return 0.5, "No image uploaded"


def classify(risk):
    if risk < 0.30:
        return "SAFE", "#00ff88", "Driving pattern is stable."
    elif risk < 0.60:
        return "MEDIUM RISK", "#ffd000", "Stay focused and avoid distractions."
    else:
        return "HIGH RISK", "#ff3b3b", "Take a break immediately. High risk detected."


def save_history(s1, s2, s3, text, img_score, risk, status, suggestion):
    row = {
        "time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "speed_variation": s1,
        "acceleration": s2,
        "steering": s3,
        "driver_condition": text,
        "image_score": round(img_score, 3),
        "risk_percent": round(risk * 100, 2),
        "status": status,
        "suggestion": suggestion
    }

    df = pd.read_csv(HISTORY_FILE)
    df = pd.concat([df, pd.DataFrame([row])], ignore_index=True)
    df.to_csv(HISTORY_FILE, index=False)


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():
    try:
        s1 = float(request.form["s1"])
        s2 = float(request.form["s2"])
        s3 = float(request.form["s3"])
        text = request.form["text"]

        txt_feat = text_feature(text)

        img_file = request.files.get("image")
        img_score, img_state = analyze_image(img_file)

        features = np.array([[s1, s2, s3, img_score, img_score, txt_feat]])

        risk = float(model.predict_proba(features)[0][1])

        status, color, suggestion = classify(risk)

        save_history(
            s1, s2, s3,
            text,
            img_score,
            risk,
            status,
            suggestion
        )

        return jsonify({
            "risk": round(risk * 100, 2),
            "status": status,
            "color": color,
            "suggestion": suggestion,
            "image_state": img_state
        })

    except Exception as e:
        return jsonify({"error": str(e)})


@app.route("/history")
def history():
    df = pd.read_csv(HISTORY_FILE)
    return df.tail(10).to_json(orient="records")


@app.route("/download_report")
def download_report():
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