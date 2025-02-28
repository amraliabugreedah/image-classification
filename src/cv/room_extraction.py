import cv2
import pytesseract
import numpy as np
import os

def extract_rooms(image_path, min_area=5000):
    """
    Extracts rooms and their text from a construction plan image.
    
    Args:
        image_path (str): Path to the input image.
        min_area (int): Minimum area for a contour to be considered a room.
    
    Returns:
        list: A list of dictionaries, one per room, containing:
              - "bounding_box": (x, y, w, h) of the room.
              - "area": area of the contour.
              - "ocr_text": text extracted from the room region.
              - "room_image": the cropped image array of the room.
    """
    # Load the original image
    image = cv2.imread(image_path)
    if image is None:
        raise FileNotFoundError(f"Unable to load image from {image_path}")
    
    # Convert to grayscale for processing
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Apply adaptive thresholding to obtain a binary image for contour detection
    thresh = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                   cv2.THRESH_BINARY_INV, 11, 2)
    
    # Find external contours
    contours, hierarchy = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    rooms = []
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < min_area:
            # Skip small contours that are likely noise
            continue
        
        # Compute bounding box for the room
        x, y, w, h = cv2.boundingRect(cnt)
        
        # Crop the room from the original image (preserving full color for OCR)
        room_img = image[y:y+h, x:x+w]
        
        # Run OCR on the cropped room image.
        # Depending on quality, you may choose to preprocess room_img before OCR.
        ocr_text = pytesseract.image_to_string(room_img)
        
        rooms.append({
            "bounding_box": (x, y, w, h),
            "area": area,
            "ocr_text": ocr_text,
            "room_image": room_img
        })
    
    return rooms

if __name__ == "__main__":
    # Example usage:
    base_dir = os.getcwd()
    # Assuming your input image is in resources/images
    image_path = os.path.join(base_dir, "resources", "images", "page_2.png")
    
    # Create a folder for saving individual room images (optional)
    output_dir = os.path.join(base_dir, "resources", "processed", "rooms")
    os.makedirs(output_dir, exist_ok=True)
    
    room_list = extract_rooms(image_path, min_area=5000)
    
    for i, room in enumerate(room_list):
        print(f"Room {i+1}:")
        print("Bounding Box:", room["bounding_box"])
        print("Area:", room["area"])
        print("OCR Text:", room["ocr_text"])
        print("-" * 40)
        
        # Optionally save each room's image
        room_img_path = os.path.join(output_dir, f"room_{i+1}.png")
        cv2.imwrite(room_img_path, room["room_image"])