import cv2
from flask import Flask, request, jsonify, render_template, Response
import numpy as np
from tensorflow.keras.models import load_model
from PIL import Image
import io
from fastai.vision.all import load_learner, PILImage
import base64
from inference_sdk import InferenceHTTPClient

app = Flask(__name__)

# Load the ripeness model (if you still want to use it) 
ripeness_model = load_model('/Users/pushpakreddy/fruit ripening detection/RipenSense/ripeness.h5')
# Load the fruit recognition model
fruit_recognition_model = load_learner('/Users/pushpakreddy/fruit ripening detection/RipenSense/fruit_recognizer.pkl')

# Initialize the Inference HTTP Client for live detection
CLIENT = InferenceHTTPClient(
    api_url="https://detect.roboflow.com",
    api_key="Ccw29Pnhy0VLHTeIb3Ul"
)

# Open camera for live detection
camera = cv2.VideoCapture(0)

# Preprocess images for the ripeness model
def preprocess_image(image, target_size):
    image = image.resize(target_size)
    image = np.array(image) / 255.0
  
    if image.shape[-1] == 1: 
        image = np.repeat(image, 3, axis=-1)
    image = np.expand_dims(image, axis=0)
    return image

# Route to render the homepage and file upload
@app.route('/')
def upload_file():
    return render_template('upload.html')

# Route to handle image upload and prediction
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

# Predict from camera frames using the Inference API
def predict_frame(frame):
    _, buffer = cv2.imencode('.jpg', frame)
    image_data = base64.b64encode(buffer).decode('utf-8')
    
    # Perform inference using the InferenceHTTPClient
    result = CLIENT.infer(image_data, model_id="fruit-ripeness-detection-zzkqi/1")
    
    # Process the inference results
    predictions = []
    for prediction in result['predictions']:
        x1 = int(prediction['x'] - prediction['width'] / 2)
        y1 = int(prediction['y'] - prediction['height'] / 2)
        x2 = int(prediction['x'] + prediction['width'] / 2)
        y2 = int(prediction['y'] + prediction['height'] / 2)
        label = prediction['class']
        confidence = prediction['confidence']
        
        predictions.append({
            'x1': x1,
            'y1': y1,
            'x2': x2,
            'y2': y2,
            'label': label,
            'confidence': confidence
        })
    
    return predictions

# Generate camera frames for live detection
def generate_frames():
    while True:
        success, frame = camera.read()
        if not success:
            break
        else:
            # Get predictions for the current frame
            predictions = predict_frame(frame)

            # Draw predictions on the frame
            for pred in predictions:
                cv2.rectangle(frame, (pred['x1'], pred['y1']), (pred['x2'], pred['y2']), (0, 255, 0), 2)
                cv2.putText(frame, f"{pred['label']} {pred['confidence']:.2f}", 
                            (pred['x1'], pred['y1'] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

            ret, buffer = cv2.imencode('.jpg', frame)
            frame = buffer.tobytes()
            
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

# Route for video feed
@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

# About and contact pages
@app.route('/about')
def about():
    return render_template('about.html')

@app.route('/contact')
def contact():
    return render_template('contact.html')

if __name__ == '__main__':
    app.run(debug=True)
