from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from skimage.feature import hog
import numpy as np
from sklearn import svm
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.metrics import accuracy_score
import joblib
from scripts.image_dataset import ImageDataset

class HOGSVM:
    def __init__(self, model_path=None):
        self.pixels_per_cell = (8, 8)
        self.cells_per_block = (2, 2)
        self.batch_size = 8
        self.num_workers = 4
        self.transform = transforms.Compose([
            transforms.Resize((128, 128)),
            transforms.ToTensor()
        ])
        self.model = make_pipeline(StandardScaler(), svm.SVC(kernel='linear'))
        if model_path:
            self.model = self.load_model(model_path)

    def load_data(self, train_image_paths, train_labels, val_image_paths, val_labels):

        train_dataset = ImageDataset(train_image_paths, train_labels, transform=self.transform)
        val_dataset = ImageDataset(val_image_paths, val_labels, transform=self.transform)

        train_loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True, num_workers=self.num_workers)
        val_loader = DataLoader(val_dataset, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers)
        return train_loader, val_loader
    
    def train_n_evaluate(self, train_image_paths, train_labels, val_image_paths, val_labels):
        train_loader, val_loader = self.load_data(train_image_paths, train_labels, val_image_paths, val_labels)
        train_hog_features = []
        train_labels_list = []
        for images, labels in train_loader:
            hog_features = self.extract_hog_features(images)
            train_hog_features.extend(hog_features)
            train_labels_list.extend(labels)
        train_hog_features = np.array(train_hog_features)
        train_labels = np.array(train_labels_list)

        self.model.fit(train_hog_features, train_labels)

        val_accuracy = self.evaluate(val_loader)
        print(f'Validation Accuracy: {val_accuracy * 100:.2f}%')
        self.save_model('../models/hog_svm.pkl')

    def extract_hog_features(self, images):
        hog_features = []
        for image in images:
            image_np = image.squeeze().numpy()
            hog_feature = hog(image_np, pixels_per_cell=self.pixels_per_cell, cells_per_block=self.cells_per_block, visualize=False)
            hog_features.append(hog_feature)
        return np.array(hog_features)

    def evaluate(self, data_loader):
        features = []
        labels = []
        for images, labels in data_loader:
            hog_features = self.extract_hog_features(images)
            features.extend(hog_features)
            labels.extend(labels)
        features = np.array(features)
        labels = np.array(labels)
        
        predictions = self.model.predict(features)
        accuracy = accuracy_score(labels, predictions)
        return accuracy
    
    def result(self):
        test_dataset = datasets.ImageFolder(root='../data/test', transform=self.transform)
        test_loader = DataLoader(test_dataset, batch_size=self.batch_size, num_workers=self.num_workers)
        test_accuracy = self.evaluate(test_loader)
        print(f'Test Accuracy: {test_accuracy * 100:.2f}%')
        return test_accuracy
    
    def save_model(self, model_path):
        joblib.dump(self.model, model_path)

    def load_model(self, model_path):
        self.model = joblib.load(model_path)
