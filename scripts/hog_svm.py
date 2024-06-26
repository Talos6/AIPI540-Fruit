import cv2
import os
import numpy as np
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from skimage.feature import hog
import joblib

ROOT = os.path.dirname(os.path.abspath(__file__))

class HOGSVM:
    def __init__(self):
        self.model = SVC(kernel='linear', probability=True)
    
    def extract_hog_features(self, image_paths):
        features = []
        for img_path in image_paths:
            # Read the image in grayscale
            img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)

            # Resize the image
            img_resized = cv2.resize(img, (64, 64))

            # Extract HOG features
            hog_features = hog(img_resized, pixels_per_cell=(8, 8), cells_per_block=(2, 2), feature_vector=True)

            features.append(hog_features)
        return np.array(features)
    
    def train_n_evaluate(self, train_image_paths, train_labels, val_image_paths, val_labels):
        # Extract train HOG features
        train_features = self.extract_hog_features(train_image_paths)

        # Train the model
        self.model.fit(train_features, train_labels)

        # Extract validation HOG features
        val_features = self.extract_hog_features(val_image_paths)

        # Model prediction on validation set
        val_predictions = self.model.predict(val_features)

        # Calculate accuracy
        accuracy = accuracy_score(val_labels, val_predictions)

        return accuracy
    
    def save_model(self):
        joblib.dump(self.model, os.path.join(ROOT, '../models/hog_svm.pkl'))

    def load_model(self):
        self.model = joblib.load(os.path.join(ROOT, '../models/hog_svm.pkl'))
