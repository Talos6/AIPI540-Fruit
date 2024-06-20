import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import timm
from scripts.image_dataset import ImageDataset

ROOT = os.path.dirname(os.path.abspath(__file__))

class TransResNet:
    def __init__(self):
        self.num_classes = 100
        self.batch_size = 8
        self.num_workers = 4
        self.epochs = 10
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
        self.model = self.build_model()
    
    def build_model(self):
        model = timm.create_model('resnet50', pretrained=True, num_classes=self.num_classes)
        model = model.to(self.device)
        return model
    
    def load_data(self, train_image_paths, train_labels, val_image_paths, val_labels):

        train_dataset = ImageDataset(train_image_paths, train_labels, transform=self.transform)
        val_dataset = ImageDataset(val_image_paths, val_labels, transform=self.transform)

        train_loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True, num_workers=self.num_workers)
        val_loader = DataLoader(val_dataset, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers)
        return train_loader, val_loader
    
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
            val_accuracy = self.evaluate(val_loader)
            print(f'Epoch {epoch + 1}/{self.epochs}, Loss: {epoch_loss:.4f}, Validation Accuracy: {val_accuracy * 100:.2f}%')

            if val_accuracy > best_val_accuracy:
                best_val_accuracy = val_accuracy
                self.save_model('../models/trans_resnet.pth')
        
        print(f'Best Validation Accuracy: {best_val_accuracy * 100:.2f}%')

    def evaluate(self, data_loader):
        self.model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for inputs, labels in data_loader:
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                outputs = self.model(inputs)
                _, predicted = torch.max(outputs, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        
        accuracy = correct / total
        return accuracy
    
    def result(self):
        test_dataset = datasets.ImageFolder(root=os.path.join(ROOT, '../data/test'), transform=self.transform)
        test_loader = DataLoader(test_dataset, batch_size=self.batch_size, num_workers=self.num_workers)
        test_accuracy = self.evaluate(test_loader)
        print(f'Test Accuracy: {test_accuracy * 100:.2f}%')
        return test_accuracy

    def save_model(self):
        torch.save(self.model.state_dict(), os.path.join(ROOT, '../models/trans_resnet.pth'))
    
    def load_model(self):
        self.model.load_state_dict(torch.load(os.path.join(ROOT, '../models/trans_resnet.pth'), map_location=self.device))
        self.model.to(self.device)
    
    def predict(self, image):
        image = self.transform(image)
        image = image.unsqueeze(0).to(self.device)

        self.model.eval()
        with torch.no_grad():
            outputs = self.model(image)
            _, predicted = torch.max(outputs, 1)
        
        return predicted.item()
    