import streamlit as st
import pandas as pd
from PIL import Image
from scripts.trans_resnet import TransResNet
from scripts.data_store import DataStore
import time

def run():
    data_store = DataStore()
    label_dict = data_store.parse_classes()

    trans_resnet = TransResNet()
    trans_resnet.load_model()

    top_half, bottom_half = st.container(), st.container()

    with top_half:
        default_image = Image.open('./data/streamlit/default.jpg')
        img_display = st.image(default_image, use_column_width=True)
        uploaded_file = st.file_uploader("Choose a image", type="jpg")
        st.header("Question: what is this fruit?")

        if uploaded_file is not None:
            image = Image.open(uploaded_file)
            img_display.image(image, use_column_width=True)
            with st.spinner('Recognizing...'):
                time.sleep(2)
            predicted_label = trans_resnet.predict(image)
            st.header(f"Answer: {label_dict.get(predicted_label)}")

    with bottom_half:
        tab1, tab2, tab3, tab4, tab5 = st.tabs(["overview", "data", "workflow", "analytics", "reference"])

        with tab1:
            st.write("## Overview")
            st.write("### Problem")
            st.write("People often encounter fruits they don't recognize and have difficulty distinguishing between similar-looking varieties. This lack of knowledge can lead to confusion and misidentification.")
            st.write("### Solution")
            st.write("developed a model capable of classifying images of 100 different types of fruits. By uploading a fruit image, the model will accurately identify and provide the correct class name, helping to resolve the issue of fruit misidentification.")
            st.write("### Usage")
            st.markdown("""
            - Take a picture of unknown fruit.
            - Upload the picture to the above file uploader, it accepts jpg format.
            - Check the answer in the header which is the identified fruit class name. 
            """)


        with tab2:
            st.write("## Data")
            st.write("### Source")
            st.write("The source is from Kaggle Fruits 100 datatsets. It consist over 14k JPEG images of different fruits. This is the original data we trained our model.")
            st.write("Here is the [link](https://www.kaggle.com/datasets/marquis03/fruits-100/data) to the original datasets.")
            st.write("### Strucutre")
            structure_data = {
                'Content': ['train/', 'val/', 'test/', 'classname.txt', 'train.csv', 'val.csv', 'test.csv'],
                'Description': ['The train dataset with 5k images, group by classes', 'The validation dataset with 4k images, group by classes', 'The test dataset with 5k images, group by labels', 'The map of class and label', 'The map of train image path and label', 'The map of validation image path and label', 'The unlabeled test image path']
            }
            st.table(pd.DataFrame(structure_data))
            st.write("### Pipeline")
            st.write("For train set and test set, the image paths and labels gathered from their corresponding CSV files. Image opened by PIL and transformed to tensor with resizing and normalizing. Data loader prepared with batch_size = 8 and num_workers = 4.")
            st.write("For test set, it already grouped up by labels so dataset are formed directly from torchvision.datasets.ImageFolder. The data loader introduced with same configuration as train and validation dataset.")


        with tab3:
            st.write("## Workflow")
            st.write("### Prior Efforts")
            st.markdown("""
            - trained all layer MobileNet model
            - trained all layer ResNet-18 model
            """)
            st.write("### Modeling")
            st.write("**Naive Approach**: The model would randomely guess the class among 100 different labels.")
            st.write("**Classical ML Approach**: Abstract Histogram of Oriented Gradients (HOG) image feature and trained a Support Vector Machine Model.")
            st.write("**Nerual Network Approach**: Integrated a pre-trained ResNet-50 Model with transfer learning the output layer.")
            st.write("### Evaluation")
            st.write("**Accuracy**: The accuracy metrics is consumed to determine the model behavior, for more details about model performance, refer to the analytics tab.")
            st.write("### Selection")
            st.write("**Classical ML has poor performance**: The classical ML approach by abstracting HOG feature and train on a SVM model performs poorly in terms of accuracy score and took hours to converge. It is because HOG features are not enough, other features like edges, dense, color still plays an important role in this classification problem. Furthermore, due to the large data set, abstract image features in CPU sequentially will costs for a quite long time to converge.")
            st.write("**NN Approach excels**: Compare with the naive approach which is random guessing along with 100 different labels, NN approach shows statistically significant improvements in the accuracy score for both validation and test.")
            st.write("### Uniqueness")
            st.markdown("""
            - **Transfer Learning**: rely on pre-trained large model as feature abstractor result in SOTA performance.
            - **Generalization**: pre-trained large model ensures we are not over-fitting, more stable to unseen data.
            - **User Friendly**: Injected model to daily workflow increased UX and participation.      
            """)

        with tab4:
            st.write("## Analytics")
            st.write("### Accuracy Score")
            accuracy_data = {
                'Model': ['trans_resnet', 'full_trained_mobilenet', 'full_trained_resnet18', 'hop_svm', 'naive approach'],
                'Validation Accuracy': [0.80, 0.69, 0.23, 0.15, 0.01]
            }
            st.table(pd.DataFrame(accuracy_data))
            st.write('### Model References')
            st.write("**full_trained_mobilenet**: [link](https://www.kaggle.com/code/literallytheone/train-all-layers-of-a-pretrained-mobilenetv2)")
            st.write("**full_trained_resnet18**: [link](https://www.kaggle.com/code/marquis03/fruit-classification-using-resnet-18)")
            st.write('### Result')
            st.write("**Validation Accuracy: 0.80**")
            st.write("**Test Accuracy: 0.78**")

        with tab5:
            st.write("## Reference")
            st.write("### Author")
            st.write("Xinyue(Yancey) Yang")
            st.write("### Repository")
            st.write("[Github link](https://github.com/Talos6/AIPI540-Fruit)")
            st.write("### Data")
            st.write("[Kaggle link](https://www.kaggle.com/datasets/marquis03/fruits-100/data)")
            st.write("### Instructions")
            code = """
                # Clone repo and ensure python and pip have installed
                git clone <repo_link> 

                # Install required libraries
                pip install -r requirements.txt

                # Run the application with trained model
                streamlit run main.py

                # Run the script to re-train models
                python setup.py
            """
            st.code(code, language='bash')

if __name__ == "__main__":
    run()