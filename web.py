from flask import Flask, request, jsonify, render_template_string, redirect, url_for, flash
from werkzeug.utils import secure_filename
import torch
from torchvision import models, transforms
from PIL import Image, UnidentifiedImageError
import json
import io

app = Flask(__name__)
app.secret_key = 'supersecretkey'  # Necessary for flash messages

# Load the metadata and model
with open('final_model_metadata.json', 'r') as f:
    metadata = json.load(f)

# Initialize the model based on metadata
model = models.resnet18(pretrained=False)
model.fc = torch.nn.Linear(metadata['fc_features'], metadata['num_classes'])

# Load the model state
model.load_state_dict(torch.load('final_model_state_dict.pth'))
model.eval()

# Set up device
device = torch.device("cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu"))
model = model.to(device)

# Define the image transformations
transform = transforms.Compose([
    transforms.Resize(metadata['input_size']),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

def predict_image(image_bytes):
    try:
        img = Image.open(io.BytesIO(image_bytes))
        img = transform(img).unsqueeze(0)  # Add batch dimension
    except UnidentifiedImageError:
        return None  # Return None if image cannot be processed

    img = img.to(device)  # Ensure tensor is on the correct device
    with torch.no_grad():
        output = model(img)
        _, predicted = torch.max(output.data, 1)
        return predicted.item()


@app.route('/', methods=['GET', 'POST'])  # Handle both GET and POST on the same route
def index():
    if request.method == 'POST':
        file = request.files['file']
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            image_bytes = file.read()
            prediction = predict_image(image_bytes)
            if prediction is not None:
                flash(f'Plant class predicted: {prediction}', 'info')
            else:
                flash('Invalid image format or corrupted image', 'error')
        else:
            flash('No file selected or unsupported file type', 'error')

    return render_template_string("""
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>Upload an Image</title>
        <style>
            body { font-family: Arial, sans-serif; background-color: #f4f4f9; color: #333; text-align: center; margin-top: 50px; }
            form { background-color: #fff; max-width: 300px; margin: auto; padding: 20px; border-radius: 8px; box-shadow: 0 0 10px rgba(0,0,0,0.1); }
            input, button { margin-top: 10px; }
            button { background-color: #4CAF50; color: white; padding: 10px 15px; border: none; border-radius: 5px; cursor: pointer; }
            button:hover { background-color: #45a049; }
            .message { color: #b00; margin: 20px auto; font-size: 16px; }
        </style>
    </head>
    <body>
        <h1>Plant Image Classifier</h1>
        <p>Select an image file of a plant, and see what it is!</p>
        {% with messages = get_flashed_messages(with_categories=true) %}
            {% if messages %}
                {% for category, message in messages %}
                    <div class="message">{{ message }}</div>
                {% endfor %}
            {% endif %}
        {% endwith %}
        <form method="post" enctype="multipart/form-data">
            <input type="file" name="file" required>
            <button type="submit">Upload Image</button>
        </form>
    </body>
    </html>
    """)


def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in {'png', 'jpg', 'jpeg', 'gif'}

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8080)
