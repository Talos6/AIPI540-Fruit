import os
import pandas as pd

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(SCRIPT_DIR, '..', 'data')
LABEL_FILE_PATH = os.path.join(SCRIPT_DIR, '..', 'data', 'classname.txt')

class DataStore:

    def parse_classes(self):
        with open(LABEL_FILE_PATH, 'r') as file:
            labels = {i: row.strip() for i, row in enumerate(file, start=0)}
        return labels

    def parse_data(self, dataset_type):
        # dataset_type is enum of train, test, val
        csv_file_path = os.path.join(DATA_DIR, f'{dataset_type}.csv')

        data = pd.read_csv(csv_file_path)
        image_paths = data['image:FILE'].tolist()
        image_paths = list(map(lambda x: os.path.join(DATA_DIR, x), image_paths))
        labels = data['category'].tolist()
        return image_paths, labels
