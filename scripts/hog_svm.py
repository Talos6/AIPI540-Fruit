import cv2
import os
import numpy as np
from scripts.data_store import DATA_DIR
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from skimage.feature import hog
import joblib

class HOGSVM:
    def __init__(self, model_path=None):
        if model_path:
            self.model = joblib.load(model_path)
        else:
            self.model = SVC(kernel='linear', probability=True)
    
    def extract_hog_features(self, image_paths):
        features = []
        for img_path in image_paths:
            img_full_path = os.path.join(DATA_DIR, img_path)
            img = cv2.imread(img_full_path, cv2.IMREAD_GRAYSCALE)
            img_resized = cv2.resize(img, (128, 128))
            hog_features = hog(img_resized, pixels_per_cell=(16, 16), cells_per_block=(4, 4), feature_vector=True)
            features.append(hog_features)
        return np.array(features)
    
    def train(self, image_paths, labels):
        train_features = self.extract_hog_features(image_paths)
        self.model.fit(train_features, labels)
    
    def evaluate(self, image_paths, labels):
        val_features = self.extract_hog_features(image_paths)
        val_predictions = self.model.predict(val_features)
        accuracy = accuracy_score(labels, val_predictions)
        return accuracy
    
    def save_model(self, model_path):
        joblib.dump(self.model, model_path)
    
    def preprocess_image(self, img_path):
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        img_resized = cv2.resize(img, (128, 128))
        hog_features = hog(img_resized, pixels_per_cell=(16, 16), cells_per_block=(4, 4), feature_vector=True)
        return hog_features
    
    def predict(self, img_path):
        features = self.preprocess_image(img_path)
        prediction = self.model.predict([features])
        return prediction[0]
    
    def predict_proba(self, img_path):
        features = self.preprocess_image(img_path)
        probabilities = self.model.predict_proba([features])
        return probabilities[0]