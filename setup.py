from scripts.data_store import DataStore
from scripts.hog_svm import HOGSVM
from scripts.trans_resnet import TransResNet

def build():
    data_store = DataStore()

    train_image_paths, train_labels = data_store.parse_data('train')
    val_image_paths, val_labels = data_store.parse_data('val')

    print('start train HOGSVM')
    hog_svm = HOGSVM()
    hog_svm.train_n_evaluate(train_image_paths, train_labels, val_image_paths, val_labels)

    trans_resnet = TransResNet()
    trans_resnet.train_n_evaluate(train_image_paths, train_labels, val_image_paths, val_labels)
    trans_resnet.result()

if __name__ == "__main__":
    build()