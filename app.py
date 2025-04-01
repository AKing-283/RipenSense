from flask import Flask, request, jsonify, render_template, Response
import numpy as np
from tensorflow.keras.models import load_model
from PIL import Image
import io
from fastai.vision.all import load_learner, PILImage
import base64


app = Flask(__name__)


ripeness_model = load_model('/Users/pushpakreddy/fruit ripening detection/RipenSense/ripeness.h5')

fruit_recognition_model = load_learner('/Users/pushpakreddy/fruit ripening detection/RipenSense/fruit_recognizer.pkl')


def preprocess_image(image, target_size):
    image = image.resize(target_size)
    image = np.array(image) / 255.0
  
    if image.shape[-1] == 1: 
        image = np.repeat(image, 3, axis=-1)
    image = np.expand_dims(image, axis=0)
    return image


@app.route('/')
def upload_file():
    return render_template('upload.html')


@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({"error": "No file uploaded"}), 400

    file = request.files['file']
    file_content = file.read()

    fruit_labels = ('Apple', 'Apricot', 'Avocado',
                    'Banana', 'Blueberry', 'Cherry', 'Fig',
                    'Grape', 'Kiwi', 'Lemon', 'Lychee', 'Mango',
                    'Orange', 'Papaya', 'Pear', 'Pineapple',
                    'Raspberry', 'Strawberry', 'Watermelon')

    try:
        img = Image.open(io.BytesIO(file_content))
        processed_image = preprocess_image(img, target_size=(224, 224))

        processed_image = np.squeeze(processed_image)
        if processed_image.shape != (224, 224, 3):
            return jsonify({"error": "Image dimensions are not 224x224x3"}), 400

        fastai_image = PILImage.create(io.BytesIO(file_content))

        def recognize_image(image):
            pred, idx, probs = fruit_recognition_model.predict(image)
            return pred, dict(zip(fruit_labels, map(float, probs)))

        fruit_name, fruit_probabilities = recognize_image(fastai_image)

        ripeness_prediction = ripeness_model.predict(np.expand_dims(processed_image, axis=0))

        ripeness_class = int(np.argmax(ripeness_prediction[0]))
        ripeness_labels = {0: "Ripe", 1: "Overipe", 2: "Unripe"}
        ripeness_status = ripeness_labels.get(ripeness_class, "Unknown")
        ripeness_probabilities = ripeness_prediction[0]

        ripeness_percentages = {
            "Ripe": round(ripeness_probabilities[0] * 100, 2),
            "Overipe": round(ripeness_probabilities[1] * 100, 2),
            "Unripe": round(ripeness_probabilities[2] * 100, 2),
        }

        encoded_image = base64.b64encode(file_content).decode('utf-8')
        image_data = f"data:image/jpeg;base64,{encoded_image}"

        return render_template(
            'result.html', 
            fruit_class=fruit_name, 
            ripeness_class=ripeness_status,
            ripeness_prediction=ripeness_prediction.tolist(), 
            fruit_probabilities=fruit_probabilities, 
            ripeness_percentages=ripeness_percentages,  
            img_path=image_data
        )

    except Exception as e:
        return jsonify({"error": f"Failed to process image: {str(e)}"}), 500


# About and contact pages
@app.route('/about')
def about():
    return render_template('about.html')

@app.route('/contact')
def contact():
    return render_template('contact.html')
    
if __name__ == "__main__":
    from os import environ
    app.run(host="0.0.0.0", port=int(environ.get("PORT", 5000)))
