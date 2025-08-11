import os
from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
import numpy as np
from PIL import Image
import io
import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms

app = FastAPI()

# Allow React frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # set to your frontend URL in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Define the EfficientNet model structure (same as in training)
def create_efficientnet_model(num_classes=4):
    """Create EfficientNet-B0 model for transfer learning"""
    model = models.efficientnet_b0(pretrained=False)  # Don't load pretrained weights
    
    # Replace the classifier
    num_features = model.classifier[1].in_features
    model.classifier = nn.Sequential(
        nn.Dropout(p=0.2),
        nn.Linear(num_features, num_classes)
    )
    
    return model

# Load the trained PyTorch model
def load_pytorch_model(model_path="road_classification_efficientnet.pth"):
    """Load the trained PyTorch model"""
    # Load the saved model data with weights_only=False (if you trust the source)
    checkpoint = torch.load(model_path, map_location=device, weights_only=False)
    
    # Get class names and number of classes
    class_names = checkpoint['class_names']
    num_classes = checkpoint['num_classes']
    
    # Create model structure
    model = create_efficientnet_model(num_classes)
    
    # Load the trained weights
    model.load_state_dict(checkpoint['model_state_dict'])
    
    # Set to evaluation mode
    model.eval()
    model = model.to(device)
    
    print(f"Model loaded successfully!")
    print(f"Classes: {class_names}")
    print(f"Device: {device}")
    
    return model, class_names

# Load model globally
loaded_model, class_names = load_pytorch_model()

# Define image transforms (same as validation transforms in training)
transform = transforms.Compose([
    transforms.Resize((224, 224)),  # EfficientNet expects 224x224
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # ImageNet normalization
])

# Your inference function
def run_inference(image):
    """Run inference on PIL image using PyTorch model"""
    # Apply transforms
    img_tensor = transform(image).unsqueeze(0)  # Add batch dimension
    img_tensor = img_tensor.to(device)
    
    # Make prediction
    with torch.no_grad():
        outputs = loaded_model(img_tensor)
        probabilities = torch.softmax(outputs, dim=1)
        confidence, predicted = torch.max(probabilities, 1)
    
    return probabilities.cpu().numpy(), predicted.cpu().numpy(), confidence.cpu().numpy()

@app.get("/")
def read_root():
    return {"message": "Welcome to the Traffic Light Prediction API"}

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    try:
        # Read uploaded image
        contents = await file.read()

        # Convert to PIL image and ensure RGB
        image = Image.open(io.BytesIO(contents)).convert("RGB")

        # Run inference using PyTorch model (no need to resize, transforms handle it)
        probabilities, predicted_class, confidence = run_inference(image)

        # Get class name from the predicted index
        predicted_class_name = class_names[predicted_class[0]]
        confidence_score = float(confidence[0])
        
        # Get all class probabilities
        class_probabilities = {}
        for i, class_name in enumerate(class_names):
            class_probabilities[class_name] = float(probabilities[0][i])

        return {
            "prediction": predicted_class_name,
            "confidence": confidence_score,
            "probabilities": class_probabilities,
            "status": "success"
        }
    
    except Exception as e:
        return {
            "error": str(e),
            "status": "error"
        }