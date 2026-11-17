import os
import json
import traceback

import numpy as np
import tensorflow as tf

from flask import Flask, request, jsonify
from flask_cors import CORS
from PIL import Image

from model_builder import build_model


# ============================================================
# Flask Configuration
# ============================================================

app = Flask(__name__)

CORS(app)


# ============================================================
# Paths
# ============================================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

MODEL_DIR = os.path.join(BASE_DIR, "model")

WEIGHTS_PATH = os.path.join(
    MODEL_DIR,
    "model.weights.h5"
)

CLASS_MAPPING_PATH = os.path.join(
    MODEL_DIR,
    "class_mapping.json"
)


# ============================================================
# Image Configuration
# ============================================================

IMG_SIZE = (224, 224)


# ============================================================
# Load Class Mapping
# ============================================================

try:
    with open(
        CLASS_MAPPING_PATH,
        "r",
        encoding="utf-8"
    ) as f:
        CLASS_MAPPING = json.load(f)

    INDEX_TO_CLASS = {
        int(index): class_name
        for class_name, index in CLASS_MAPPING.items()
    }

    print("Class mapping loaded:")
    print(INDEX_TO_CLASS)

except Exception as e:

    print("WARNING: Could not load class_mapping.json")
    print(e)

    CLASS_MAPPING = {
        "glioma": 0,
        "healthy": 1,
        "meningioma": 2,
        "pituitary": 3
    }

    INDEX_TO_CLASS = {
        0: "glioma",
        1: "healthy",
        2: "meningioma",
        3: "pituitary"
    }


# ============================================================
# Model
# ============================================================

model = None
model_load_error = None


def load_trained_model():

    global model
    global model_load_error

    try:

        print()
        print("=" * 60)
        print("Loading Brain Tumor CAD model...")
        print("=" * 60)

        print("Weights file:")
        print(WEIGHTS_PATH)

        # Check weights file
        if not os.path.exists(WEIGHTS_PATH):

            raise FileNotFoundError(
                f"Model weights not found: {WEIGHTS_PATH}"
            )

        # Build the exact architecture
        model = build_model()

        print()
        print("Model architecture created successfully.")

        print(
            "Total parameters:",
            model.count_params()
        )

        # Load trained Kaggle weights
        model.load_weights(
            WEIGHTS_PATH
        )

        print()
        print("Trained weights loaded successfully.")

        print(
            "Model name:",
            model.name
        )

        print(
            "Input shape:",
            model.input_shape
        )

        print(
            "Output shape:",
            model.output_shape
        )

        # ----------------------------------------------------
        # Warm-up prediction
        # ----------------------------------------------------

        dummy_image = np.zeros(
            (1, 224, 224, 3),
            dtype=np.float32
        )

        model.predict(
            dummy_image,
            verbose=0
        )

        print(
            "Model warm-up successful."
        )

        model_load_error = None

        print()
        print("=" * 60)
        print("MODEL READY")
        print("=" * 60)
        print()

    except Exception as e:

        model = None
        model_load_error = str(e)

        print()
        print("=" * 60)
        print("MODEL LOADING FAILED")
        print("=" * 60)

        print(str(e))

        traceback.print_exc()

        print("=" * 60)
        print()


# Load trained model when server starts
load_trained_model()


# ============================================================
# Image Preprocessing
# ============================================================

def preprocess_image(image):

    # Convert image to RGB
    image = image.convert("RGB")

    # Resize to training size
    image = image.resize(
        IMG_SIZE
    )

    # Convert to NumPy array
    image_array = np.asarray(
        image,
        dtype=np.float32
    )

    # Add batch dimension
    image_array = np.expand_dims(
        image_array,
        axis=0
    )

    # ConvNeXt preprocessing
    image_array = tf.keras.applications.convnext.preprocess_input(
        image_array
    )

    return image_array


# ============================================================
# Health Check API
# ============================================================

@app.route(
    "/api/health",
    methods=["GET"]
)
def health():

    return jsonify({

        "status": "ok",

        "model_loaded": (
            model is not None
        ),

        "model_name": (
            model.name
            if model is not None
            else None
        ),

        "classes": INDEX_TO_CLASS,

        "load_error": model_load_error
    })


# ============================================================
# Prediction API
# ============================================================

@app.route(
    "/api/predict",
    methods=["POST"]
)
def predict():

    try:

        # ----------------------------------------------------
        # Check model
        # ----------------------------------------------------

        if model is None:

            return jsonify({

                "error":
                    "Model is not loaded.",

                "details":
                    model_load_error

            }), 500

        # ----------------------------------------------------
        # Check uploaded image
        # ----------------------------------------------------

        if "image" not in request.files:

            return jsonify({

                "error":
                    "No image uploaded."

            }), 400

        file = request.files["image"]

        if file.filename == "":

            return jsonify({

                "error":
                    "No image selected."

            }), 400

        # ----------------------------------------------------
        # Open image
        # ----------------------------------------------------

        image = Image.open(
            file.stream
        )

        # ----------------------------------------------------
        # Preprocess image
        # ----------------------------------------------------

        processed_image = preprocess_image(
            image
        )

        # ----------------------------------------------------
        # Model prediction
        # ----------------------------------------------------

        predictions = model.predict(
            processed_image,
            verbose=0
        )[0]

        # Find highest probability
        predicted_index = int(
            np.argmax(predictions)
        )

        predicted_class = INDEX_TO_CLASS.get(
            predicted_index,
            str(predicted_index)
        )

        confidence = float(
            predictions[predicted_index]
        )

        # ----------------------------------------------------
        # Per-class probabilities
        # ----------------------------------------------------

        probabilities = {}

        for index, probability in enumerate(
            predictions
        ):

            class_name = INDEX_TO_CLASS.get(
                index,
                str(index)
            )

            probabilities[class_name] = round(
                float(probability) * 100,
                2
            )

        # ----------------------------------------------------
        # Return result
        # ----------------------------------------------------

        return jsonify({

            "success": True,

            "prediction": {

                "class":
                    predicted_class,

                "label":
                    predicted_class.capitalize(),

                "confidence":
                    round(
                        confidence * 100,
                        2
                    )
            },

            "probabilities":
                probabilities

        })

    except Exception as e:

        traceback.print_exc()

        return jsonify({

            "error":
                "Prediction failed.",

            "details":
                str(e)

        }), 500


# ============================================================
# Start Server
# ============================================================

if __name__ == "__main__":

    print()
    print("=" * 60)
    print("Brain Tumor CAD Backend")
    print("=" * 60)
    print("Server: http://localhost:5000")
    print("Health: http://localhost:5000/api/health")
    print("=" * 60)
    print()

    app.run(
        host="0.0.0.0",
        port=5000,
        debug=False
    )