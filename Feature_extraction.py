from torchvision import models, transforms
from PIL import Image
import torch
import os

# Load the pre-trained ResNet model
resnet_model = models.resnet50(pretrained=True)

# Set the model to evaluation mode
resnet_model.eval()

# Define the image transformations - resizing, normalization, and others as needed
transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# Load the image
image_path = '/Users/Anto/PycharmProjects/OpenCV/Progetto/Principessa.png'
image = Image.open(image_path)

# Apply the transformations to the image
transformed_image = transform(image)

# Add an extra batch dimension since pytorch treats all images as batches
batch = transformed_image.unsqueeze(0)

# Move the input and model to GPU for speed if available
if torch.cuda.is_available():
    batch = batch.to('cuda')
    resnet_model.to('cuda')

with torch.no_grad():
    # Get the features from the image
    features = resnet_model(batch)

# Convert features to a numpy array
features_numpy = features.cpu().numpy()

# Output the numpy array of features
features_numpy.shape, os.path.getsize(image_path)
