
import os
import json
import numpy as np
import requests
import tensorflow as tf
from flask import Flask, render_template, request
from werkzeug.utils import secure_filename

# -------------------------
# Config
# -------------------------
SINGLE_MODE = True   # 🔄 Set True = only top-1 prediction, False = top-3

app = Flask(__name__)
APP_ROOT = os.path.dirname(os.path.abspath(__file__))
UPLOAD_FOLDER = os.path.join(APP_ROOT, "static", "uploads")
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER

# -------------------------
# Load model & DB
# -------------------------
MODEL_PATH = os.path.join(APP_ROOT, "models", "plant_disease_recog_model_pwp.keras")
model = tf.keras.models.load_model(MODEL_PATH)

LABELS = [
 'Apple___Apple_scab',
 'Apple___Black_rot',
 'Apple___Cedar_apple_rust',
 'Apple___healthy',
 'Background_without_leaves',
 'Blueberry___healthy',
 'Cherry___Powdery_mildew',
 'Cherry___healthy',
 'Corn___Cercospora_leaf_spot Gray_leaf_spot',
 'Corn___Common_rust',
 'Corn___Northern_Leaf_Blight',
 'Corn___healthy',
 'Grape___Black_rot',
 'Grape___Esca_(Black_Measles)',
 'Grape___Leaf_blight_(Isariopsis_Leaf_Spot)',
 'Grape___healthy',
 'Orange___Haunglongbing_(Citrus_greening)',
 'Peach___Bacterial_spot',
 'Peach___healthy',
 'Pepper,_bell___Bacterial_spot',
 'Pepper,_bell___healthy',
 'Potato___Early_blight',
 'Potato___Late_blight',
 'Potato___healthy',
 'Raspberry___healthy',
 'Soybean___healthy',
 'Squash___Powdery_mildew',
 'Strawberry___Leaf_scorch',
 'Strawberry___healthy',
 'Tomato___Bacterial_spot',
 'Tomato___Early_blight',
 'Tomato___Late_blight',
 'Tomato___Leaf_Mold',
 'Tomato___Septoria_leaf_spot',
 'Tomato___Spider_mites Two-spotted_spider_mite',
 'Tomato___Target_Spot',
 'Tomato___Tomato_Yellow_Leaf_Curl_Virus',
 'Tomato___Tomato_mosaic_virus',
 'Tomato___healthy'
]

with open(os.path.join(APP_ROOT, "plant_disease.json"), "r", encoding="utf-8") as f:
    DISEASE_DB = json.load(f)

# -------------------------
# Weather API
# -------------------------
API_KEY = "b7604be9b87762925dc679afb0ca4881"     #os.getenv("OWM_API_KEY")  # set your API key here
WEATHER_URL = "https://api.openweathermap.org/data/2.5/weather"

def get_weather(city):
    if not API_KEY:
        return None
    try:
        params = {"q": city, "appid": API_KEY, "units": "metric"}
        resp = requests.get(WEATHER_URL, params=params)
        data = resp.json()
        if resp.status_code == 200:
            return {
                "temp": data["main"]["temp"],
                "humidity": data["main"]["humidity"],
                "desc": data["weather"][0]["description"],
            }
    except Exception as e:
        print("Weather API error:", e)
    return None

# -------------------------
# Model Prediction
# -------------------------
def predict_disease(img_path):
    img = tf.keras.utils.load_img(img_path, target_size=(160, 160))
    img_array = tf.keras.utils.img_to_array(img)
    img_array = np.array([img_array])

    preds = model.predict(img_array, verbose=0)[0]

    if SINGLE_MODE:
        # only top-1 prediction
        idx = int(np.argmax(preds))
        result = [(LABELS[idx], float(preds[idx]))]
    else:
        # top-3 predictions
        top_indices = preds.argsort()[-3:][::-1]
        result = [(LABELS[i], float(preds[i])) for i in top_indices]

    print("DEBUG >> Raw preds:", preds)
    print("DEBUG >> Result:", result)

    return result

# -------------------------
# Remedy Selector
# -------------------------
def pick_remedy(disease_name, weather):
    db_entry = DISEASE_DB.get(disease_name, {})
    remedies = db_entry.get("remedy", {})

    if not remedies:
        return "General field hygiene and crop rotation are advised."

    if weather:
        if weather["humidity"] > 70:
            return remedies.get("humid", remedies.get("default", "No remedy found."))
        elif weather["temp"] > 30:
            return remedies.get("dry", remedies.get("default", "No remedy found."))
    return remedies.get("default", "No remedy found.")

# -------------------------
# Routes
# -------------------------
@app.route("/")
def index():
    return render_template("index.html")

@app.route("/analyze", methods=["POST"])
def analyze():
    lang = request.form.get("lang", "en")
    city = request.form.get("city", "")
    file = request.files.get("image")

    if not file:
        return "No file uploaded", 400

    filename = secure_filename(file.filename)
    save_path = os.path.join(app.config["UPLOAD_FOLDER"], filename)
    file.save(save_path)

    predictions = predict_disease(save_path)
    weather = get_weather(city)

    enriched = []
    for label, prob in predictions:
        db_entry = DISEASE_DB.get(label, {})
        enriched.append({
            "label": label,
            "name": db_entry.get("name", {}).get(lang, label),
            "why": db_entry.get("why", {}).get(lang, "Info not available."),
            "fertilizer": db_entry.get("fertilizer", "General fertilizer advice."),
            "remedy": pick_remedy(label, weather),
            "prob": round(prob * 100, 2)
        })

    return render_template(
        "result.html",
        predictions=enriched,
        weather=weather,
        lang=lang,
        image_file=filename
    )

# -------------------------
# Main
# -------------------------
if __name__ == "__main__":
    app.run(debug=True)
