# AIPI540-Fruit

## Description
The project aims to classify different types of fruits and provide a user interface that accepting any fruit image to be predicted.

## Author
Xinyue(Yancey) Yang

## Data source
[Link to Kaggle dataset](https://www.kaggle.com/datasets/marquis03/fruits-100/data)

## Model
**Classical ML**: A pipeline which reads image greyscal, abstract HOG features, and fitted into a SVM model.

**Neural Network**: A pretrained ResNet50 with modified output layer. 

## Instruction
To run the code in this repository, please follow these steps:

1. Clone the repository to your local machine:
    ```
    git clone https://github.com/Talos6/AIPI540-Fruit.git
    ```

2. Navigate to the project directory:
    ```
    cd AIPI540-Fruit
    ```

3. Install the required dependencies:
    ```
    pip install -r requirements.txt
    ```

4. Run the application:
    ```
    streamlit run main.py
    ```

5. Re-train the model:
    ```
    python setup.py
    ```
