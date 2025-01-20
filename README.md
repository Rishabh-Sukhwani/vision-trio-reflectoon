# Cartoon Matching Project

This project identifies the cartoon character that most closely resembles a given input image of a human face. It uses MediaPipe for facial landmark detection and compares features against a predefined set of cartoon images.

---

## **Setup Instructions**

### **Directory Structure**
Ensure the following directory structure is set up:

```
project-directory/
|-- cartoon_images/         # Directory containing cartoon images
|   |-- cartoon_1.png       # Cartoon image 1
|   |-- cartoon_2.png       # Cartoon image 2
|   |-- ...
|   |-- cartoon_15.png      # Cartoon image 15
|
|-- cartoon_features.npy    # Precomputed features of the cartoon images
|-- input_image.jpg         # Input image to be matched
|-- main.py                 # Python script with the main logic
```

---

## **Naming Conventions**

1. **Cartoon Images**:
   - Name your cartoon images in the format `cartoon_X.jpg`, where `X` is a number between `1` and `15`.
   - Example: `cartoon_1.jpg`, `cartoon_2.jpg`, ..., `cartoon_15.jpg`.

2. **Input Image**:
   - Name the input image `input_image.jpg`, or update the `input_image_path` variable in the script to point to your image file.

3. **Features File**:
   - The precomputed features for the cartoon images must be stored in a file named `cartoon_features.npy` in the project root directory.

---

## **How It Works**
1. The script reads the `input_image.jpg` and extracts facial landmarks using MediaPipe.
2. It calculates the Euclidean distance between the facial landmarks of the input image and precomputed features of the cartoon images.
3. The cartoon image with the smallest distance is selected as the closest match.
4. The matched cartoon image is displayed.

---

## **Running the Script**
1. Install the necessary dependencies:
   ```bash
   pip install opencv-python mediapipe numpy pillow scikit-learn
   ```
2. Place your input image in the project root or update the `input_image_path` variable.
3. Run the script:
   ```bash
   python preprocess.py
   ```
4. Run:
    ```bash
    python capture.py
    ```

---

## **Dependencies**
- OpenCV
- MediaPipe
- scikit-learn
- NumPy

---

## **Notes**
- Ensure the cartoon images have similar dimensions or aspect ratios for better visualization.
- If you encounter errors with file paths, double-check the directory structure and file names.
- To add more cartoon images, update the naming convention (e.g., `cartoon_16.jpg`, `cartoon_17.jpg`, etc.) and recompute the `cartoon_features.npy` file.

