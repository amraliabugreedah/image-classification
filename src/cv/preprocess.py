import cv2
import os
from datetime import datetime
import base64
import json


def preprocess_image(image_path, output_path=None):
    """
    Preprocess the input image for further processing.

    Steps:
      1. Load the image.
      2. Convert the image to grayscale.
      3. Apply Gaussian blur to reduce noise.
      4. Use adaptive thresholding to create a binary image.

    Args:
      image_path (str): Path to the input image.
      output_path (str, optional): If provided, save the preprocessed image here.

    Returns:
      numpy.ndarray: The preprocessed image.
    """
    # Load the image from the provided path
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError(f"Unable to load image from {image_path}")

    # Convert the image to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Apply Gaussian blur to reduce noise
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)

    # Apply adaptive thresholding to obtain a binary image
    preprocessed = cv2.adaptiveThreshold(
        blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 2
    )

    # If an output path is specified, save the preprocessed image
    if output_path:
        cv2.imwrite(output_path, preprocessed)

    return preprocessed


if __name__ == "__main__":

    # List of image names to preprocess

    resources_dir = os.path.join(os.getcwd(), "resources", "images", "base64toimg")
    image_names = [
        os.path.splitext(f)[0] for f in os.listdir(resources_dir) if f.endswith(".png")
    ]

    base64_list = []

    for image_name in image_names:
        # Define the path for the input image in the resources folder
        image_filename = f"{image_name}.png"  # Change to your actual image filename
        image_path = os.path.join(resources_dir, image_filename)

        # Define an output directory and filename for the preprocessed image
        output_dir = os.path.join(
            os.getcwd(), "resources", "processed", "test_tokens", "img"
        )
        os.makedirs(output_dir, exist_ok=True)
        output_path = os.path.join(output_dir, f"preprocessed_{image_name}.png")

        # Run preprocessing
        preprocessed_image = preprocess_image(image_path, output_path)
        print("Preprocessing complete. Saved preprocessed image at:", output_path)

        base_64_output_dir = os.path.join(
            os.getcwd(), "resources", "processed", "test_tokens", "basea64"
        )

        os.makedirs(base_64_output_dir, exist_ok=True)
        # Generate a timestamp for the output filename
        timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
        base64_output_path = os.path.join(
            base_64_output_dir, f"preprocessed_{timestamp}.json"
        )

        # Convert the preprocessed image to base64
        _, buffer = cv2.imencode(".png", preprocessed_image)
        base64_image = base64.b64encode(buffer).decode("utf-8")
        base64_list.append(base64_image)

    # Save the base64 string to a file
    with open(base64_output_path, "w") as f:
      json.dump(base64_list, f)

    print("Base64 encoding complete. Saved base64 list at:", base64_output_path)
