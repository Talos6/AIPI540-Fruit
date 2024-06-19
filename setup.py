from scripts.data_store import DataStore
from scripts.hog_svm import HOGSVM

if __name__ == "__main__":
    data_store = DataStore()

    train_image_paths, train_labels = data_store.parse_data('train')
    val_image_paths, val_labels = data_store.parse_data('val')

    hog_svm = HOGSVM()
    hog_svm.train(train_image_paths, train_labels)
    print(hog_svm.evaluate(val_image_paths, val_labels))
    hog_svm.save_model('/models/hog_svm.pkl')