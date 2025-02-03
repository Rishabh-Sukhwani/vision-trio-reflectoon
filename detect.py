import base64
import cv2
import numpy as np
from mediapipe import solutions
from sklearn.metrics.pairwise import euclidean_distances
from similar import find_similar_face

# Initialize MediaPipe FaceMesh
mp_face_mesh = solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(static_image_mode=True)

# Directory containing cartoon images and features
CARTOON_DIR = "cartoon_images/"
FEATURES_FILE = "cartoon_features.npy"

# Load precomputed cartoon features
cartoon_features = np.load(FEATURES_FILE)
cartoon_names = [f"cartoon_{i}.png" for i in range(1, 5)]  # Assuming cartoon names are in order


# Function to extract facial landmarks from an image
def extract_landmarks(image):
    results = face_mesh.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    if results.multi_face_landmarks:
        # Extract landmarks as a flattened array
        landmarks = []
        for lm in results.multi_face_landmarks[0].landmark:
            landmarks.append([lm.x, lm.y, lm.z])
        return np.array(landmarks).flatten()
    return None


# Load the input image
input_image_path = "input_image_2.jpg"  # Replace with your image filename
input_image = cv2.imread(input_image_path)




# Function to encode the image
def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode("utf-8")


# Path to your image
#image_path = r"C:\Users\risha\OneDrive\Desktop\Rishabh\projects\hack\input_image_5.jpg"
#image_path = r"C:\Users\risha\OneDrive\Pictures\Camera Roll\WIN_20250203_11_11_51_Pro.jpg"
image_path = r"C:\Users\risha\OneDrive\Pictures\Camera Roll\WIN_20250203_11_15_24_Pro.jpg"

find_similar_face(image_path=image_path)