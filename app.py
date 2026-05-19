import os
import cv2
import faiss
import pickle
import base64
import numpy as np
from flask import Flask, render_template, request, jsonify
import tensorflow as tf
from ultralytics import YOLO
from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras.applications.efficientnet import preprocess_input
from sklearn.preprocessing import normalize

# Force CPU processing
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

app = Flask(__name__)

# ==========================================
# 1. CONFIGURATION & THRESHOLDS
# ==========================================
INDEX_FILE = "cow_muzzle.index"
MAPPING_FILE = "cow_mapping.pkl"
YOLO_MODEL_PATH = "best.pt"

IMG_SIZE = 224
SIMILARITY_THRESHOLD = 0.85
YOLO_CONF_THRESHOLD = 0.80
BLUR_THRESHOLD = 100

# ==========================================
# 2. LOAD MODELS (Runs once on startup)
# ==========================================
print("Loading AI Models... Please wait.")
yolo_model = YOLO(YOLO_MODEL_PATH)

index = faiss.read_index(INDEX_FILE)
with open(MAPPING_FILE, 'rb') as f:
    cow_mapping = pickle.load(f)

eff_model = EfficientNetB0(weights='imagenet', include_top=False, pooling='avg')
print("AI Models Loaded Successfully!")


# ==========================================
# 3. HELPER FUNCTIONS
# ==========================================
def is_blurry(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    variance = cv2.Laplacian(gray, cv2.CV_64F).var()
    return variance < BLUR_THRESHOLD


def identify_crop(crop_img):
    img = cv2.resize(crop_img, (IMG_SIZE, IMG_SIZE))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    img_array = np.expand_dims(img, axis=0)
    img_array = preprocess_input(img_array)

    embedding = eff_model.predict(img_array, verbose=0)
    embedding = normalize(embedding, norm='l2').astype('float32')

    distances, indices = index.search(embedding, 1)
    return indices[0][0], distances[0][0]


# ==========================================
# 4. FLASK ROUTES
# ==========================================
@app.route('/')
def index_page():
    return render_template('index.html')


@app.route('/process_live_frame', methods=['POST'])
def process_live_frame():
    try:
        # 1. Receive the Base64 image from the live webcam
        data = request.json['image']
        header, encoded = data.split(",", 1)

        # 2. Decode the Base64 string back into an OpenCV image
        nparr = np.frombuffer(base64.b64decode(encoded), np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        # 3. Detect Muzzles with YOLO
        results = yolo_model(img, verbose=False)
        
        detections = []

        for box in results[0].boxes:
            conf = float(box.conf)
            if conf < YOLO_CONF_THRESHOLD:
                continue

            x1, y1, x2, y2 = map(int, box.xyxy[0])
            crop = img[y1:y2, x1:x2]

            if crop.size == 0 or is_blurry(crop):
                continue

            # 4. Identify the Cropped Muzzle
            best_match_id, similarity_score = identify_crop(crop)

            # 5. Draw Bounding Box (Known or Unknown)
            if similarity_score > SIMILARITY_THRESHOLD:
                predicted_name = cow_mapping[best_match_id]
                label = f"{predicted_name} ({similarity_score * 100:.1f}%)"
                color = (0, 255, 0) # Green for known
                detections.append(predicted_name)
            else:
                label = f"Unknown ({similarity_score * 100:.1f}%)"
                color = (0, 0, 255) # Red for unknown
                detections.append("Unknown")

            cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
            cv2.putText(img, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)

        # Determine Status text
        muzzle_count = len(results[0].boxes)
        if muzzle_count == 0:
            status = "No muzzle detected"
        elif len(detections) == 0:
            status = "Muzzle detected (too blurry or low confidence)"
        else:
            status = f"Detected: {', '.join(detections)}"

        # 6. Encode the processed image back to Base64 to send to the browser
        _, buffer = cv2.imencode('.jpg', img)
        processed_base64 = base64.b64encode(buffer).decode('utf-8')

        return jsonify({
            'image': 'data:image/jpeg;base64,' + processed_base64,
            'status': status
        })

    except Exception as e:
        print(f"Frame Error: {e}")
        return jsonify({'error': str(e)}), 500


if __name__ == '__main__':
    # Use debug=False for live camera to prevent heavy reloading overhead
    app.run(host='0.0.0.0', port=5000, debug=False)