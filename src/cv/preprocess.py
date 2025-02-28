import cv2
import os


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
    # Define the path for the input image in the resources folder
    resources_dir = os.path.join(os.getcwd(), "resources", "images")
    image_filename = "sample_plan.png"  # Change to your actual image filename
    image_path = os.path.join(resources_dir, image_filename)

    # Define an output directory and filename for the preprocessed image
    output_dir = os.path.join(os.getcwd(), "resources", "processed")
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, "preprocessed_sample_plan.png")

    # Run preprocessing
    preprocessed_image = preprocess_image(image_path, output_path)
    print("Preprocessing complete. Saved preprocessed image at:", output_path)
