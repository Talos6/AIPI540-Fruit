import os
import pandas as pd

class DataStore:
    def parse_classes(self):
        with open('../data/classname.txt', 'r') as file:
            labels = {i: row.strip() for i, row in enumerate(file, start=0)}
        return labels

    def parse_data(self, dataset_type):
        # dataset_type is enum of train, test, val
        csv_file_path = os.path.join('../data/', f'{dataset_type}.csv')

        data = pd.read_csv(csv_file_path)
        image_paths = data['image:FILE'].tolist()
        image_paths = list(map(lambda x: os.path.join('../data/', x), image_paths))
        labels = data['category'].tolist()
        return image_paths, labels
