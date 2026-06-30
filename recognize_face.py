import cv2
import pickle

from insightface.app import FaceAnalysis
from sklearn.metrics.pairwise import cosine_similarity

# ----------------------------------
# Load ArcFace
# ----------------------------------

app = FaceAnalysis(
    name="buffalo_l",
    providers=["CPUExecutionProvider"]
)

app.prepare(ctx_id=0)


# ----------------------------------
# Load Database
# ----------------------------------

def load_database(database_path):

    with open(database_path, "rb") as f:
        database = pickle.load(f)

    print(f"Loaded {len(database)} embeddings")

    return database


# ----------------------------------
# Recognition Function
# ----------------------------------

def recognize_face(
    face_image,
    database,
    threshold=0.55
):

    if face_image is None:
        raise ValueError("Face image is None")

    faces = app.get(face_image)

    if len(faces) != 1:
        return {
            "label": "No Face",
            "similarity": 0
        }

    query_embedding = faces[0].embedding.reshape(1, -1)

    best_similarity = -1
    best_label = "unknown"

    for item in database:

        db_embedding = item["embedding"].reshape(1, -1)

        similarity = cosine_similarity(
            query_embedding,
            db_embedding
        )[0][0]

        if similarity > best_similarity:

            best_similarity = similarity
            best_label = item["label"]

    if best_similarity < threshold:
        best_label = "unknown"

    return {
        "label": best_label,
        "similarity": float(best_similarity)
    }


# ----------------------------------
# Test
# ----------------------------------

if __name__ == "__main__":

    database = load_database(
        "data/embeddings/face_database.pkl"
    )

    image = cv2.imread("test.jpg")

    result = recognize_face(
        image,
        database
    )

    print(result)