import cv2
import pytesseract
from PIL import Image
import numpy as np
import os

def two_pass_preprocessing(image_path, ocr_output_path=None, line_output_path=None):
    """
    Perform a two-pass preprocessing on a construction plan image:
      1) OCR Pass: Minimal processing for text extraction.
      2) Line Pass: Aggressive threshold & morphological ops for structure/line detection.
    
    Args:
        image_path (str): Path to the input image.
        ocr_output_path (str, optional): If provided, save the OCR-friendly image here.
        line_output_path (str, optional): If provided, save the line-detection image here.
    
    Returns:
        dict: A dictionary containing:
          - "ocr_text": The text extracted by Tesseract.
          - "ocr_image": The OCR-friendly (minimally processed) image as a numpy array.
          - "line_image": The line-detection (aggressively processed) image as a numpy array.
    """

    # 1) Load & Convert to Grayscale
    original = cv2.imread(image_path)
    if original is None:
        raise FileNotFoundError(f"Could not load image at {image_path}")

    gray = cv2.cvtColor(original, cv2.COLOR_BGR2GRAY)

    # -----------------------------
    # 2) Pass #1: OCR Extraction
    #    (Minimal or no thresholding)
    # -----------------------------
    # Option A: Use the raw grayscale
    # Option B: Use Otsu threshold if needed

    # For demonstration, let's just keep it grayscale
    ocr_image = gray.copy()

    # Save the OCR-friendly image if desired
    if ocr_output_path:
        cv2.imwrite(ocr_output_path, ocr_image)

    # Convert to PIL for Tesseract
    pil_ocr_image = Image.fromarray(ocr_image)
    ocr_text = pytesseract.image_to_string(pil_ocr_image)

    # -----------------------------
    # 3) Pass #2: Line Detection
    #    (Aggressive threshold & morphological ops)
    # -----------------------------
    # Let's do Otsu's threshold for a crisp black-and-white image
    _, line_thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)

    # Morphological closing to connect broken lines
    # Adjust kernel size if lines are thicker or thinner
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    line_closed = cv2.morphologyEx(line_thresh, cv2.MORPH_CLOSE, kernel, iterations=1)

    line_image = line_closed

    # Save the line-detection image if desired
    if line_output_path:
        cv2.imwrite(line_output_path, line_image)

    return {
        "ocr_text": ocr_text,
        "ocr_image": ocr_image,
        "line_image": line_image
    }


if __name__ == "__main__":
    # Example usage if run directly
    base_dir = os.getcwd()
    image_path = os.path.join(base_dir, "resources", "images", "page_2.png")
    
    # Create an output folder for processed images
    processed_dir = os.path.join(base_dir, "resources", "processed")
    os.makedirs(processed_dir, exist_ok=True)
    
    ocr_output = os.path.join(processed_dir, "ocr_pass.png")
    line_output = os.path.join(processed_dir, "line_pass.png")
    
    result = two_pass_preprocessing(
        image_path,
        ocr_output_path=ocr_output,
        line_output_path=line_output
    )
    
    print("OCR Text:")
    print(result["ocr_text"])
    print("OCR image saved to:", ocr_output)
    print("Line-detection image saved to:", line_output)