from flask import Flask, request, jsonify
from PIL import Image
import io
import tensorflow as tf
import numpy as np
import os
app = Flask(__name__)

# Load TensorFlow Lite model
model_path = "Backend/detect.tflite"

interpreter = tf.lite.Interpreter(model_path=model_path)

interpreter.allocate_tensors()

def run_inference(image):
    # Preprocess the image for the model
    # This depends on the model's expected input size and format
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    # Resize image and convert to expected format
    input_shape = input_details[0]['shape']
    image = image.resize((input_shape[1], input_shape[2]))
    input_data = np.expand_dims(image, axis=0)

    # Run the model
    interpreter.set_tensor(input_details[0]['index'], input_data)
    interpreter.invoke()

    # Extract bounding box data
    bbox_data = interpreter.get_tensor(output_details[0]['index'])
    return bbox_data

@app.route('/infer', methods=['POST'])
def infer():
    if request.method == 'POST':
        # Check if an image is part of the request
        if 'image' not in request.files:
            return jsonify({'error': 'No image part in the request'}), 400

        file = request.files['image']
        if file.filename == '':
            return jsonify({'error': 'No image selected for uploading'}), 400

        if file:
            # Convert the file to an PIL Image
            image = Image.open(io.BytesIO(file.read()))

            # Run inference
            bbox = run_inference(image)

            # Return the bounding box data
            return jsonify({'bounding_box': bbox.tolist()})
    
    return jsonify({'error': 'Invalid request method'}), 405

if __name__ == '__main__':
    app.run(debug=True)