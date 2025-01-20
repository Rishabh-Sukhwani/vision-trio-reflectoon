import cv2
import numpy as np
from mediapipe import solutions

# Directory containing cartoon images
CARTOON_DIR = "cartoon_images"
OUTPUT_FEATURES_FILE = "cartoon_features.npy"

# Initialize MediaPipe FaceMesh
mp_face_mesh = solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(static_image_mode=True)

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

# Process all cartoon images
cartoon_features = []
cartoon_names = []

for i in range(1, 5):  # Assuming cartoon images are named cartoon_1.jpg, ..., cartoon_15.jpg
    image_path = f"{CARTOON_DIR}/cartoon_{i}.png"
    image = cv2.imread(image_path)
    if image is None:
        print(f"Image not found: {image_path}")
        continue

    # Extract landmarks
    features = extract_landmarks(image)
    if features is not None:
        cartoon_features.append(features)
        cartoon_names.append(f"cartoon_{i}.jpg")

# Save extracted features
np.save(OUTPUT_FEATURES_FILE, np.array(cartoon_features))
print(f"Saved cartoon features to {OUTPUT_FEATURES_FILE}")
