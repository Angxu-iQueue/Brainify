from flask import Flask, request, render_template, redirect, url_for
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
from PIL import Image

# Initialize Flask app
app = Flask(__name__)

# Load your trained model
model = load_model('model.h5')

# Define the class labels
class_labels = [ 'glioma', 'meningioma', 'no tumor', 'pituitary']

# Home route
@app.route('/')
def home():
    return render_template('index.html')

# Route to handle image upload and prediction
@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return redirect(url_for('home'))
    
    file = request.files['file']

    if file and file.filename != '':
        # Preprocess the uploaded image
        img = Image.open(file).convert('RGB')
        img = img.resize((224, 224))  # Resize to the input shape expected by the model
        img = np.expand_dims(img, axis=0)  # Add batch dimension
        img = np.array(img) / 255.0  # Normalize the image

        # Make prediction
        prediction = model.predict(img)
        predicted_class = np.argmax(prediction, axis=1)[0]
        predicted_label = class_labels[predicted_class]

        return render_template('result.html', label=predicted_label)

    return redirect(url_for('home'))

if __name__ == '__main__':
    app.run(debug=True)
