import streamlit as st
from PIL import Image

def render():
    st.title('Fruits')

    top_half, bottom_half = st.container(), st.container()

    with top_half:
        st.header("fruits?")
        default_image = Image.open('/data/streamlit/default.jpg')
        img_display = st.image(default_image, use_column_width=True)
        uploaded_file = st.file_uploader("Choose a JPG image", type="jpg")

        if uploaded_file is not None:
            image = Image.open(uploaded_file)
            img_display.image(image, use_column_width=True)

    with bottom_half:
        tab1, tab2, tab3, tab4 = st.tabs(["data", "analytics", "workflow", "reference"])

        with tab1:
            st.write("## Kaggle Dataset")
            st.write("Here is a [link](https://www.kaggle.com/) to the Kaggle dataset.")
            st.write("Description of the data...")

        with tab2:
            st.write("## Analytics")
            st.write("Validation Accuracy: 95%")
            st.write("Test Accuracy: 93%")

        with tab3:
            st.write("## Workflow")
            st.write("**Problem**: Describe the problem here.")
            st.write("**Modeling**: Explain the modeling process here.")
            st.write("**Uniqueness**: Describe what makes this approach unique.")

        with tab4:
            st.write("## Reference")
            st.write("Here is a [link](https://github.com/) to the GitHub repository.")

if __name__ == "__main__":
    render()