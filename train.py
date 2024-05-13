import os
import torch
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.optim as optim
import json
from tqdm import tqdm

# Transformations for the input data
transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Function to filter directories based on image count
def filter_directories(base_path, min_count):
    filtered_classes = {}
    for class_dir in os.listdir(base_path):
        class_path = os.path.join(base_path, class_dir)
        if os.path.isdir(class_path):
            num_images = len([name for name in os.listdir(class_path) if os.path.isfile(os.path.join(class_path, name))])
            if num_images >= min_count:
                filtered_classes[class_dir] = num_images
    return filtered_classes

# Load the datasets with ImageFolder
train_base_path = 'plant_images/train'
test_base_path = 'plant_images/test'

# Filter classes
train_classes = filter_directories(train_base_path, 10)
test_classes = filter_directories(test_base_path, 5)

# Get common classes with enough images
common_classes = set(train_classes.keys()).intersection(set(test_classes.keys()))

# Ensure there are common classes with enough images
if not common_classes:
    raise RuntimeError("No valid classes with the required number of images found.")

# Filter function for datasets
def is_valid_file(path):
    return path.split(os.sep)[-2] in common_classes

# Initialize datasets
training_dataset = datasets.ImageFolder(train_base_path, transform=transform, is_valid_file=is_valid_file)
testing_dataset = datasets.ImageFolder(test_base_path, transform=transform, is_valid_file=is_valid_file)

# Define dataloaders
train_loader = DataLoader(training_dataset, batch_size=32, shuffle=True)
test_loader = DataLoader(testing_dataset, batch_size=32, shuffle=True)

# Initialize and configure the model
model = models.resnet18(pretrained=True)
num_features = model.fc.in_features
model.fc = nn.Linear(num_features, len(common_classes))

# Move model to the best available device
device = torch.device("cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu"))
model = model.to(device)

# Loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

def save_model_state_dict(model, path="model_state_dict.pth", metadata_path="model_metadata.json"):
    """Save the model state dictionary and metadata to files."""
    torch.save(model.state_dict(), path)
    metadata = {
        'num_classes': num_classes,
        'input_size': (256, 256),  # Input size for resizing images
        'model_architecture': 'resnet18',
        'fc_features': num_features,
        'class_to_index': training_dataset.class_to_idx  # Save class to index mapping
    }
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f)


def train_model(model, criterion, optimizer, num_epochs=30):
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        running_corrects = 0
        progress_bar = tqdm(train_loader, desc=f'Epoch {epoch+1}/{num_epochs}', leave=False)

        for inputs, labels in progress_bar:
            inputs = inputs.to(device)
            labels = labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            _, preds = torch.max(outputs, 1)
            loss_item = loss.item() * inputs.size(0)
            corrects_item = torch.sum(preds == labels.data)
            running_loss += loss_item
            running_corrects += corrects_item

            progress_bar.set_postfix(loss=loss_item / len(inputs), accuracy=corrects_item.float() / len(inputs))

        epoch_loss = running_loss / len(training_dataset)
        epoch_acc = running_corrects.float() / len(training_dataset)
        
        print(f'Completed Epoch {epoch+1}/{num_epochs}. Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')
    
    # Save model and metadata at the end of training
    save_model_state_dict(model, "final_model_state_dict.pth", "final_model_metadata.json")

def evaluate_model(model):
    model.eval()
    running_corrects = 0
    
    for inputs, labels in test_loader:
        inputs = inputs.to(device)
        labels = labels.to(device)

        with torch.no_grad():
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
        
        running_corrects += torch.sum(preds == labels.data)
    
    acc = running_corrects.float() / len(testing_dataset)
    print(f'Test Accuracy: {acc:.4f}')

# Train the model
train_model(model, criterion, optimizer, num_epochs=3000)

# Evaluate the model
evaluate_model(model)
