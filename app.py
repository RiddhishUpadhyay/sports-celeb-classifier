import io
import base64
import cv2
import numpy as np

from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import StreamingResponse

from predict_image import predict_image

# ----------------------------------
# Create FastAPI App
# ----------------------------------

app = FastAPI(
    title="Sports Celebrity Classifier API",
    description="Detect and recognize Indian cricketers in uploaded images.",
    version="1.0.0"
)


# ----------------------------------
# Home Endpoint
# ----------------------------------

@app.get("/")
def home():

    return {
        "message": "Sports Celebrity Classifier API is running!"
    }


# ----------------------------------
# Prediction Endpoint
# ----------------------------------

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    # Validate uploaded file
    if not file.content_type.startswith("image/"):
        raise HTTPException(
            status_code=400,
            detail="Please upload a valid image."
        )

    contents = await file.read()

    print(f"Received {len(contents)} bytes")

    image = cv2.imdecode(
        np.frombuffer(contents, np.uint8),
        cv2.IMREAD_COLOR
    )

    if image is None:
        raise HTTPException(
            status_code=400,
            detail="Unable to read image."
        )

    # Run prediction
    output_image, predictions = predict_image(image)

    if len(predictions) == 0:
        raise HTTPException(
            status_code=400,
            detail="No recognizable faces were found in the image."
        )

    # Print predictions in terminal
    # print("\nPrediction Summary")
    # print("=" * 40)

    # for prediction in predictions:

    #     print(
    #         f"Face {prediction['face_id']} | "
    #         f"Label: {prediction['label']} | "
    #         f"Similarity: {prediction['similarity']:.4f}"
    #     )

    # Encode image
    success, encoded_image = cv2.imencode(".jpg", output_image)

    if not success:
        raise HTTPException(
            status_code=500,
            detail="Failed to encode output image."
        )

    # Return image
    image_base64 = base64.b64encode(encoded_image.tobytes()).decode("utf-8")

    return {
        "predictions": predictions,
        "image": image_base64
    }
