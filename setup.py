from scripts.data_store import DataStore
from scripts.trans_resnet import TransResNet

if __name__ == "__main__":
    data_store = DataStore()

    train_image_paths, train_labels = data_store.parse_data('train')
    val_image_paths, val_labels = data_store.parse_data('val')

    trans_resnet = TransResNet()
    trans_resnet.train_n_evaluate(train_image_paths, train_labels, val_image_paths, val_labels)