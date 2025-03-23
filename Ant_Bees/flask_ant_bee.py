from flask import Flask, request, render_template
import requests
import json
from PIL import Image
import io

app = Flask(__name__)

# Azure ML Endpoint URL
AZURE_ENDPOINT = "http://41576a85-d534-45ee-b1ff-be2c2946630c.eastus2.azurecontainer.io/score"  # Replace with your actual endpoint URL

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        if 'file' not in request.files:
            return render_template('index.html', prediction="No file uploaded.")
        
        file = request.files['file']
        if file.filename == '':
            return render_template('index.html', prediction="No file selected.")
        
        image = Image.open(file.stream).convert("RGB")
        img_bytes = io.BytesIO()
        image.save(img_bytes, format='JPEG')
        img_bytes = img_bytes.getvalue()
        
        # Send image to Azure ML endpoint
        headers = {'Content-Type': 'application/json'}
        payload = json.dumps({"image": list(img_bytes)})
        response = requests.post(AZURE_ENDPOINT, data=payload, headers=headers)
        
        try:
            response_json = response.json()  # Ensure we parse JSON safely
            if isinstance(response_json, dict):  # Check if it's a dictionary
                prediction = response_json.get("prediction", "Unknown")
            else:
                prediction = f": {response_json}"
        except requests.exceptions.JSONDecodeError:
            prediction = f"Invalid JSON response: {response.text}"
        
        return render_template('index.html', prediction=f"Predicted: {prediction}")
    
    return render_template('index.html', prediction=None)

if __name__ == '__main__':
    app.run(debug=True)