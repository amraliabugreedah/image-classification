import cv2
import numpy as np
import pytesseract
import os

def detect_rooms_advanced(image_path, output_dir=None, min_area=5000):
    """
    Detects rooms in a floor plan by extracting horizontal and vertical lines, then combining them
    to form enclosed polygons. Each polygon is assumed to be a room, which is cropped, OCR'd,
    and optionally saved as a separate image.
    
    Args:
        image_path (str): Path to the floor plan image.
        output_dir (str, optional): Directory to save extracted room images. If None, rooms aren't saved.
        min_area (int): Minimum area for a contour to be considered a valid room.
    
    Returns:
        list of dict: Each dict contains:
            - "bounding_box": (x, y, w, h) bounding box of the room
            - "area": contour area
            - "ocr_text": text extracted from the room
            - "room_image": cropped BGR image of the room
            - "saved_path": file path where the room image was saved (None if not saved)
    """

    # 1) Load the original image
    original = cv2.imread(image_path)
    if original is None:
        raise FileNotFoundError(f"Could not load image at {image_path}")
    
    gray = cv2.cvtColor(original, cv2.COLOR_BGR2GRAY)

    # 2) Binarize (Otsu’s threshold)
    #    We'll invert so that walls are white (255) on a black background (0).
    _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)
    # At this point, lines/walls should appear in white.

    # 3) Extract Horizontal Lines
    horizontal = thresh.copy()
    horizontal_size = max(10, horizontal.shape[1] // 100)  # Heuristic for kernel width
    horiz_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (horizontal_size, 1))
    horizontal = cv2.erode(horizontal, horiz_kernel, iterations=1)
    horizontal = cv2.dilate(horizontal, horiz_kernel, iterations=1)

    # 4) Extract Vertical Lines
    vertical = thresh.copy()
    vertical_size = max(10, vertical.shape[0] // 100)  # Heuristic for kernel height
    vert_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, vertical_size))
    vertical = cv2.erode(vertical, vert_kernel, iterations=1)
    vertical = cv2.dilate(vertical, vert_kernel, iterations=1)

    # 5) Combine horizontal & vertical lines
    combined_lines = cv2.bitwise_or(horizontal, vertical)

    # 6) Morphological Closing to fill small gaps at intersections
    close_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    closed = cv2.morphologyEx(combined_lines, cv2.MORPH_CLOSE, close_kernel, iterations=2)

    # 7) Find contours in the closed image
    contours, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Create output directory if saving
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    rooms = []
    room_index = 0

    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < min_area:
            # Skip small contours
            continue

        x, y, w, h = cv2.boundingRect(cnt)
        room_crop = original[y:y+h, x:x+w]  # Crop from the original BGR image

        # OCR on the cropped region (convert to grayscale or threshold if needed)
        ocr_gray = cv2.cvtColor(room_crop, cv2.COLOR_BGR2GRAY)
        ocr_text = pytesseract.image_to_string(ocr_gray)

        # Save each room image if output_dir is provided
        saved_path = None
        if output_dir:
            room_filename = f"room_{room_index+1}.png"
            saved_path = os.path.join(output_dir, room_filename)
            cv2.imwrite(saved_path, room_crop)

        rooms.append({
            "bounding_box": (x, y, w, h),
            "area": area,
            "ocr_text": ocr_text.strip(),
            "room_image": room_crop,
            "saved_path": saved_path
        })
        room_index += 1

    return rooms

if __name__ == "__main__":
    # Example usage
    test_image_path = "resources/images/page_2.png"  # Adjust path as needed
    output_rooms_dir = "resources/processed/rooms/advanced"

    detected_rooms = detect_rooms_advanced(
        test_image_path,
        output_dir=output_rooms_dir,
        min_area=1000
    )

    print(f"Found {len(detected_rooms)} rooms.")
    for i, room in enumerate(detected_rooms):
        print(f"Room {i+1}:")
        print("  Bounding Box:", room["bounding_box"])
        print("  Area:", room["area"])
        print("  OCR Text:", room["ocr_text"])
        print("  Saved Path:", room["saved_path"])
        print("-" * 50)