import torch
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.optim as optim

# Transformations for the input data
transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Load the datasets with ImageFolder
training_dataset = datasets.ImageFolder('plant_images/train', transform=transform)
testing_dataset = datasets.ImageFolder('plant_images/test', transform=transform)

# Define the dataloaders
train_loader = DataLoader(training_dataset, batch_size=32, shuffle=True)
test_loader = DataLoader(testing_dataset, batch_size=32, shuffle=True)

model = models.resnet18(pretrained=True)
num_features = model.fc.in_features
num_classes = len(training_dataset.classes)  # number of plant types

# Replace the fully connected layer
model.fc = nn.Linear(num_features, num_classes)

# Move model to GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu"))
model = model.to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

def train_model(model, criterion, optimizer, num_epochs=10):
    for epoch in range(num_epochs):
        model.train()  # Set model to training mode
        running_loss = 0.0
        running_corrects = 0
        
        for inputs, labels in train_loader:
            inputs = inputs.to(device)
            labels = labels.to(device)
            
            optimizer.zero_grad()
            
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            _, preds = torch.max(outputs, 1)
            running_loss += loss.item() * inputs.size(0)
            running_corrects += torch.sum(preds == labels.data)
        
        epoch_loss = running_loss / len(training_dataset)
        epoch_acc = running_corrects.double() / len(training_dataset)
        
        print(f'Epoch {epoch+1}/{num_epochs} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')

# Train the model
train_model(model, criterion, optimizer, num_epochs=30)

def evaluate_model(model):
    model.eval()  # Set model to evaluate mode
    running_corrects = 0
    
    for inputs, labels in test_loader:
        inputs = inputs.to(device)
        labels = labels.to(device)

        with torch.no_grad():
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
        
        running_corrects += torch.sum(preds == labels.data)
    
    acc = running_corrects.double() / len(testing_dataset)
    print(f'Test Accuracy: {acc:.4f}')

# Evaluate the model
evaluate_model(model)
