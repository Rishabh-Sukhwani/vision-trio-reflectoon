import cv2
import numpy as np
from mediapipe import solutions
from sklearn.metrics.pairwise import euclidean_distances

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

if input_image is None:
    print(f"Error: Unable to load image from {input_image_path}")
else:
    # Extract features from the input image
    input_features = extract_landmarks(input_image)

    if input_features is not None:
        # Match features with cartoon features
        distances = euclidean_distances([input_features], cartoon_features)
        closest_index = np.argmin(distances)
        matched_cartoon = cartoon_names[closest_index]
        print(f"Matched Cartoon: {matched_cartoon} (Distance: {distances[0][closest_index]:.2f})")

        # Load the matched cartoon image
        matched_cartoon_path = f"{CARTOON_DIR}/{matched_cartoon}"
        matched_cartoon_image = cv2.imread(matched_cartoon_path)

        if matched_cartoon_image is None:
            print(f"Error: Unable to load cartoon image from {matched_cartoon_path}")
        else:
            # Display the matched cartoon image
            cv2.imshow("Matched Cartoon", matched_cartoon_image)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
    else:
        print("No face detected in the input image.")
