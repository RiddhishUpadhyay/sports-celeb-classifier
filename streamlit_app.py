import streamlit as st
import requests
import pandas as pd
import base64
from PIL import Image
from io import BytesIO

# ----------------------------------
# Page Config
# ----------------------------------

st.set_page_config(
    page_title="Sports Celebrity Classifier",
    layout="wide"
)

st.title("Sports Celebrity Classifier")

st.write(
    "Upload an image containing one or more faces. "
    "The application will detect and recognize known athlets."
)
st.write(
    "User can upload images and app will recognize images of Virat Kohli, Rohit Sharma, MS Dhoni and Jasprit Bumrah. "
    "Other athletes would be labelled as unkowns."
)
# ----------------------------------
# Upload Image
# ----------------------------------

uploaded_file = st.file_uploader(
    "Choose an image",
    type=["jpg", "jpeg", "png", "webp"]
)

if uploaded_file is not None:

    # st.image(uploaded_file, caption="Uploaded Image", use_container_width=True)
    uploaded_image = Image.open(uploaded_file)

    _, center, _ = st.columns([1, 1, 1])
    
    with center:
        predict = st.button("Predict",use_container_width=True)

    if predict:

        with st.spinner("Running prediction..."):
            uploaded_file.seek(0)
            file_bytes = uploaded_file.read()

            response = requests.post(
                "http://127.0.0.1:8000/predict",
                files={
                    "file": (
                        uploaded_file.name,
                        file_bytes,
                        uploaded_file.type
                    )
                }
            )

        if response.status_code == 200:

            result = response.json()

            predictions = result["predictions"]

            image_bytes = base64.b64decode(result["image"])

            image = Image.open(BytesIO(image_bytes))

            st.success("Prediction Complete!")

            # st.subheader("Annotated Image")

            # st.image(
            #     image,
            #     use_container_width=True
            # )
            col1, col2 = st.columns(2)

            with col1:
                st.subheader("Uploaded Image")
                st.image(uploaded_image, width=350)

            with col2:
                st.subheader("Prediction")
                st.image(image,width=350)

            st.subheader("Detected Faces")

            df = pd.DataFrame(predictions)
            # ----------------------------------
            # Display Labels
            # ----------------------------------

            LABEL_MAP = {
                "kohli": "Virat Kohli",
                "rohit": "Rohit Sharma",
                "msd": "MS Dhoni",
                "bumrah": "Jasprit Bumrah",
                "unknown": "Unknown",
                "No Face": "No Face"
            }

            df["label"] = df["label"].map(
                lambda x: LABEL_MAP.get(x, x)
            )

            df["similarity"] = (
                df["similarity"] * 100
            ).round(2)

            df.rename(
                columns={
                    "face_id": "Face ID",
                    "label": "Celebrity",
                    "similarity": "Confidence (%)"
                },
                inplace=True
            )

            # st.table(df)
            st.dataframe(df, use_container_width=True, hide_index=True)

        else:

            st.error(response.json()["detail"])