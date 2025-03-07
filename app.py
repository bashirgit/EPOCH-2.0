from flask import Flask, render_template, request, jsonify
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import io
import os

app = Flask(__name__)

# Load the model
model = models.mobilenet_v2(weights=None)  # Initialize without pre-trained weights
model.classifier[1] = nn.Linear(in_features=1280, out_features=9)  # Ensure it matches the trained model's classes
model.load_state_dict(torch.load("garbage_classifier.pth", map_location=torch.device("cpu")))
model.eval()

# Define transformations for input images
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Define class labels
class_labels = ['Cardboard', 'Food Organics', 'Glass', 'Metal', 'Miscellaneous Trash', 'Paper', 'Plastic', 'Textile Trash', 'Vegetation']

@app.route('/')
def index():
    return render_template("index.html")

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({"error": "No file provided"}), 400

    file = request.files['file']
    if file.filename == '':
        return jsonify({"error": "No selected file"}), 400

    try:
        image = Image.open(io.BytesIO(file.read())).convert("RGB")
        image = transform(image).unsqueeze(0)  # Add batch dimension

        with torch.no_grad():
            output = model(image)
            predicted_class = torch.argmax(output, dim=1).item()
        
        return jsonify({"prediction": class_labels[predicted_class]})
    
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)
