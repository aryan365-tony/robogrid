import base64
import cv2
import numpy as np
from flask import Flask, request, jsonify, render_template
import keras._tf_keras.keras as keras
import sys
import io

# Set the default encoding to utf-8
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

app = Flask(__name__)
#https://drive.google.com/file/d/1BJ2mCPcvQglxDFR8E260DiFBmVvZNY7u/view?usp=sharing

def download_model(destination):
    URL = f"https://drive.google.com/uc?id=1BJ2mCPcvQglxDFR8E260DiFBmVvZNY7u"
    response = requests.get(URL)
    if response.status_code == 200:
        with open(destination, 'wb') as f:
            f.write(response.content)
    else:
        raise Exception(f"Failed to download file from Google Drive: {response.status_code}")

try:
    model_file_id = 'YOUR_FILE_ID'  # Replace with your actual file ID
    download_model('fresh_model.keras')
    model = keras.models.load_model('fresh_model.keras')
except Exception as e:
    print(f"Error loading model: {str(e)}")
    model = None


vegetables = [
    "banana", "beans broad", "beans cluster", "beans haricot", "beetroot",
    "bitter guard", "bottle guard", "brinjal long", "brinjal[purple]", "cabbage",
    "capsicum green", "carrot", "cauliflower", "chilli green", "colocasia arvi",
    "corn", "cucumber", "drumstick", "garlic", "ginger", "ladies finger",
    "lemons", "Onion red", "potato", "sweet potato", "tomato", "Zuchini"
]

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Extract and decode image data
        data = request.json
        image_data = data['image'].split(',')[1]  # Remove data:image/jpeg;base64,

        # Decode the base64 string into a NumPy array
        nparr = np.frombuffer(base64.b64decode(image_data), np.uint8)

        # Convert the NumPy array into an OpenCV image (BGR format)
        image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        # Convert to RGB format
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Resize the image to the expected input size of the model (e.g., 225x225)
        image = cv2.resize(image, (225, 225))

        # Normalize the image
        image = image.astype('float32') / 255.0  # Normalize to [0, 1]
        image = np.expand_dims(image, axis=0)  # Add batch dimension

        # Make a prediction using the model
        predictions = model.predict(image)
        predicted_class = np.argmax(predictions, axis=1)[0]

        # Map the class index to the vegetable label
        prediction_label = vegetables[predicted_class]

        return jsonify({'prediction': prediction_label})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)
