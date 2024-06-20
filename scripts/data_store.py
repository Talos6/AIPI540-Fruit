import os
import pandas as pd

ROOT = os.path.dirname(os.path.abspath(__file__))

class DataStore:
    """
    Universal class to parse data from txt and csv files.
    """
    def parse_classes(self):
        with open(os.path.join(ROOT, '../data/classname.txt'), 'r') as file:
            labels = {i: row.strip() for i, row in enumerate(file, start=0)}
        return labels

    def parse_data(self, dataset_type):
        """
        Parse data from csv files. dataset_type can be 'train', 'val', or 'test'.
        """
        csv_file_path = os.path.join(ROOT, '../data/', f'{dataset_type}.csv')
        data = pd.read_csv(csv_file_path)
        image_paths = data['image:FILE'].tolist()
        image_paths = list(map(lambda x: os.path.join(ROOT, '../data/', x), image_paths))
        labels = data['category'].tolist()
        return image_paths, labels
