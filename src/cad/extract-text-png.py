import cv2
import pytesseract
import os
import time
import concurrent.futures


def extract_text_from_right_column(
    image_path: str, right_column_percentage: float = 0.25
) -> str:
    """
    Extracts text from the right column of an image without knowing its resolution in advance.

    The function determines the image resolution at runtime and crops the rightmost portion
    based on the provided percentage.

    Args:
        image_path (str): Path to the image file.
        right_column_percentage (float): Fraction of the image width to consider as the right column.
                                         Default is 0.25 (25% of the width).

    Returns:
        str: Extracted text from the cropped region.
    """
    # Load the image
    image = cv2.imread(image_path)
    if image is None:
        raise FileNotFoundError(f"Unable to load image from {image_path}")

    # Get the image dimensions: height, width, and channels
    height, width, _ = image.shape
    # print(f"Image dimensions: width={width}, height={height}")

    # Calculate the starting x-coordinate for the right column
    x_start = int(width * (1 - right_column_percentage))

    # Crop the image: taking all rows and columns from x_start to the end
    cropped = image[:, x_start:]

    # Convert the cropped image to grayscale (improves OCR accuracy)
    gray = cv2.cvtColor(cropped, cv2.COLOR_BGR2GRAY)

    # Optional: apply thresholding to binarize the image (adjust threshold as needed)
    _, thresh = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY)

    # Use pytesseract to perform OCR on the cropped image
    custom_config = (
        r"--oem 3 --psm 6"  # You can adjust these options based on your layout
    )
    text = pytesseract.image_to_string(thresh, config=custom_config)

    # Check if the extracted text contains the phrase "drawing title" else it should fail
    if "drawing title" not in text.lower():
        raise ValueError("The image does not contain 'drawing title'.")

    # Extract text between "Drawing Title" and "Drawing"
    start_index = text.lower().find("drawing title")
    end_index = text.lower().find("drawing", start_index + len("drawing title"))
    extracted_section = ""

    # Check if the extracted between drawing title is there else it should fail
    if start_index != -1 and end_index != -1:
        extracted_section = text[start_index + len("drawing title") : end_index].strip()

    else:
        print("Could not find the section between 'Drawing Title' and 'Drawing'.")

    return extracted_section


if __name__ == "__main__":
    image_folder = "<Path to folder>/images/base64toimg"  # Path to your floorplan image
    result_list = []

    def process_image(filename):
        if filename.endswith(".png"):
            image_path = os.path.join(image_folder, filename)
            extracted_text = extract_text_from_right_column(
                image_path, right_column_percentage=0.25
            )

            return extracted_text

    start_time = time.time()
    # Use ThreadPoolExecutor to process images concurrently
    with concurrent.futures.ThreadPoolExecutor() as executor:
        futures = [
            executor.submit(process_image, filename)
            for filename in os.listdir(image_folder)
        ]
        for future in concurrent.futures.as_completed(futures):
            result_list.append(future.result())
    end_time = time.time()
    elapsed_time = end_time - start_time
    print(result_list)
    print(f"Time taken to process images: {elapsed_time:.2f} seconds")
