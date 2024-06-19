import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from PIL import Image
import os
import timm

class CustomImageDataset(Dataset):
    def __init__(self, image_paths, labels, transform=None):
        self.image_paths = image_paths
        self.labels = labels
        self.transform = transform
    
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        image = Image.open(self.image_paths[idx]).convert('RGB')
        label = self.labels[idx]
        if self.transform:
            image = self.transform(image)
        return image, label

class TransResNet:
    def __init__(self, model_path=None):
        self.num_classes = 100
        self.batch_size = 8
        self.num_workers = 4
        self.epochs = 10
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.model = self.build_model()
        if model_path:
            self.model = self.load_model(model_path)
    
    def build_model(self):
        model = timm.create_model('resnet50', pretrained=True, num_classes=self.num_classes)
        model = model.to(self.device)
        return model
    
    def load_data(self, train_image_paths, train_labels, val_image_paths, val_labels):
        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])

        train_dataset = CustomImageDataset(train_image_paths, train_labels, transform=transform)
        val_dataset = CustomImageDataset(val_image_paths, val_labels, transform=transform)

        train_loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True, num_workers=self.num_workers)
        val_loader = DataLoader(val_dataset, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers)
        return train_loader, val_loader
    
    def validation_accuracy(self, val_loader):
        self.model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                outputs = self.model(inputs)
                _, predicted = torch.max(outputs, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        
        accuracy = correct / total
        return accuracy
    
    def train_n_evaluate(self, train_image_paths, train_labels, val_image_paths, val_labels):
        train_loader, val_loader = self.load_data(train_image_paths, train_labels, val_image_paths, val_labels)
        
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(self.model.parameters(), lr=3e-4)
        best_val_accuracy = 0.0
        
        for epoch in range(self.epochs):
            self.model.train()
            running_loss = 0.0
            for inputs, labels in train_loader:
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                optimizer.zero_grad()
                outputs = self.model(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
                running_loss += loss.item()

            epoch_loss = running_loss / len(train_loader)
            val_accuracy = self.validation_accuracy(val_loader)
            print(f'Epoch {epoch + 1}/{self.epochs}, Loss: {epoch_loss:.4f}, Validation Accuracy: {val_accuracy * 100:.2f}%')

            if val_accuracy > best_val_accuracy:
                best_val_accuracy = val_accuracy
                self.save_model('../models/trans_resnet.pth')
        
        print(f'Best Validation Accuracy: {best_val_accuracy * 100:.2f}%')

    
    def save_model(self, model_path):
        os.makedirs(os.path.dirname(model_path), exist_ok=True)
        torch.save(self.model.state_dict(), model_path)
        print(f'Model saved to {model_path}')
    
    def load_model(self, model_path):
        self.model.load_state_dict(torch.load(model_path))
        self.model.to(self.device)
        print(f'Model loaded from {model_path}')
    