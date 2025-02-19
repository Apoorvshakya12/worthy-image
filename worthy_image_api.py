# Flask web application for image processing
# This application allows users to upload an image, processes the image using the val_extract function,
# and returns the analysis results as a JSON response.

from flask import Flask, request, jsonify
import datetime
import pytz
import numpy as np
import cv2 as cv
from worthy_image import val_extract  # Assuming val_extract is implemented as provided

# Initialize the Flask application
app = Flask(__name__)

# Define the route for processing an image
@app.route('/process_image', methods=['POST'])
def process_image():
    try:
        # Check if the 'image' field is part of the uploaded files
        if 'image' not in request.files:
            # Return an error if no image is uploaded
            return jsonify({'error': 'No image uploaded'}), 400
        
        # Read the image file from the request
        image = request.files['image']
        image_bytes = image.read()  # Convert image to bytes
        
        # Call the val_extract function to process the image
        result = val_extract(image_bytes)
        
        # Check if the result from val_extract is valid
        if result:
            result['filename'] = image.filename  # Add the original filename to the result
            # Return the processed result as a JSON response
            return jsonify(result)
        else:
            # Return an error if the image processing fails
            return jsonify({'error': 'Image processing failed'}), 500
    
    except Exception as e:
        # Catch any unexpected errors and return them in the response
        return jsonify({'error': str(e)}), 500
    
# Flask application execution
if __name__ == '__main__':
    # Run the Flask app in debug mode for easier development and troubleshooting
    app.run(debug=True)
