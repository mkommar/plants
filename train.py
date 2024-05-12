import torch
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.optim as optim

# Transformations for the input data
transform = transforms.Compose([
    transforms.Resize(256),  # Resize the images to 256x256
    transforms.CenterCrop(224),  # Crop the images to 224x224 for model compatibility
    transforms.ToTensor(),  # Convert images to PyTorch tensor format
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # Normalize images
])

# Load the datasets with ImageFolder
training_dataset = datasets.ImageFolder('plant_images/train', transform=transform)
testing_dataset = datasets.ImageFolder('plant_images/test', transform=transform)

# Define the dataloaders
train_loader = DataLoader(training_dataset, batch_size=32, shuffle=True)
test_loader = DataLoader(testing_dataset, batch_size=32, shuffle=True)

# Initialize the model: ResNet18 pretrained on ImageNet
model = models.resnet18(pretrained=True)
num_features = model.fc.in_features  # Get the number of input features of the final layer
num_classes = len(training_dataset.classes)  # Determine the number of classes from the dataset

# Replace the fully connected layer for our specific class number
model.fc = nn.Linear(num_features, num_classes)

# Move model to the best available device
device = torch.device("cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu"))
model = model.to(device)

# Define the loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

def train_model(model, criterion, optimizer, num_epochs=30):
    for epoch in range(num_epochs):
        model.train()  # Set the model to training mode
        running_loss = 0.0
        running_corrects = 0
        
        for inputs, labels in train_loader:
            inputs = inputs.to(device).to(torch.float32)  # Ensure inputs are float32
            inputs = inputs.to(device)
            labels = labels.to(device)
            
            optimizer.zero_grad()  # Zero the parameter gradients
            
            # Forward pass
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()  # Backpropagation
            optimizer.step()  # Optimize
            
            # Statistics
            _, preds = torch.max(outputs, 1)
            running_loss += loss.item() * inputs.size(0)
            running_corrects += torch.sum(preds == labels.data)
        
        epoch_loss = running_loss / len(training_dataset)
        epoch_acc = running_corrects / len(training_dataset)
        
        print(f'Epoch {epoch+1}/{num_epochs} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')

def evaluate_model(model):
    model.eval()  # Set the model to evaluation mode
    running_corrects = 0
    
    for inputs, labels in test_loader:
        inputs = inputs.to(device).to(torch.float32)  # Ensure input is float32
        inputs = inputs.to(device)
        labels = labels.to(device)

        with torch.no_grad():  # No gradients needed
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
        
        running_corrects += torch.sum(preds == labels.data)
    
    acc = running_corrects / len(testing_dataset)
    print(f'Test Accuracy: {acc:.4f}')

# Train the model
train_model(model, criterion, optimizer)

# Evaluate the model
evaluate_model(model)
